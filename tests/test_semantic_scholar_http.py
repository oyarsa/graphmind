"""Tests for Semantic Scholar HTTP retry helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self, cast

import aiohttp
import pytest

from paper.semantic_scholar import http as semantic_scholar_http
from paper.semantic_scholar.http import fetch_json_with_retries


class _FakeResponse:
    """Minimal aiohttp-like response object for unit tests."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool:
        return False

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    """Minimal aiohttp-like session object for unit tests."""

    def __init__(self, outcomes: list[_FakeResponse | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: Mapping[str, Any]) -> _FakeResponse:
        self.calls.append((url, dict(params)))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch sleep to avoid real delays and capture the backoff schedule."""
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(semantic_scholar_http.asyncio, "sleep", _fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_fetch_json_with_retries_returns_on_first_success() -> None:
    """Return immediately when the first request succeeds."""
    session = _FakeSession([_FakeResponse({"ok": True})])

    result = await fetch_json_with_retries(
        cast(aiohttp.ClientSession, session),
        params={"a": 1},
        url="https://example.org",
        max_tries=3,
    )

    assert result == {"ok": True}
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_fetch_json_with_retries_retries_on_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry on client errors and then return a successful response."""
    delays = _patch_sleep(monkeypatch)
    session = _FakeSession([
        aiohttp.ClientError("boom"),
        _FakeResponse({"ok": "recovered"}),
    ])

    result = await fetch_json_with_retries(
        cast(aiohttp.ClientSession, session),
        params={"a": 1},
        url="https://example.org",
        max_tries=2,
    )

    assert result == {"ok": "recovered"}
    assert len(session.calls) == 2
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_fetch_json_with_retries_raises_after_max_tries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise the final retryable exception after exhausting retries."""
    delays = _patch_sleep(monkeypatch)
    session = _FakeSession([TimeoutError(), TimeoutError()])

    with pytest.raises(asyncio.TimeoutError):
        await fetch_json_with_retries(
            cast(aiohttp.ClientSession, session),
            params={"a": 1},
            url="https://example.org",
            max_tries=2,
        )

    assert len(session.calls) == 2
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_fetch_json_with_retries_retries_when_validator_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry when the response validator raises a retryable exception."""
    delays = _patch_sleep(monkeypatch)
    session = _FakeSession([
        _FakeResponse({"ok": False}, status=500),
        _FakeResponse({"ok": True}),
    ])
    validator_calls = 0

    def _validator(response: aiohttp.ClientResponse) -> None:
        nonlocal validator_calls
        validator_calls += 1
        if response.status >= 500:
            raise aiohttp.ClientError("server unavailable")

    result = await fetch_json_with_retries(
        cast(aiohttp.ClientSession, session),
        params={"a": 1},
        url="https://example.org",
        max_tries=2,
        validate_response=_validator,
    )

    assert result == {"ok": True}
    assert validator_calls == 2
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_fetch_json_with_retries_rejects_non_positive_max_tries() -> None:
    """Validate retry configuration before making requests."""
    session = _FakeSession([])

    with pytest.raises(ValueError, match="max_tries must be positive"):
        await fetch_json_with_retries(
            cast(aiohttp.ClientSession, session),
            params={},
            url="https://example.org",
            max_tries=0,
        )
