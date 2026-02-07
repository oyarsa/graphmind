"""Tests for backend rate limiting utilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from paper.backend.rate_limiter import RateLimiter, _parse_rule, get_remote_address


def _request(
    *,
    path: str = "/mind/evaluate",
    client_host: str = "127.0.0.1",
    forwarded_for: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=client_host),
        url=SimpleNamespace(path=path),
    )


def test_get_remote_address_ignores_forwarded_header_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without trusted-proxy mode, use direct client address."""
    monkeypatch.delenv("XP_TRUST_PROXY_HEADERS", raising=False)
    request = _request(client_host="10.0.0.1", forwarded_for="203.0.113.9")
    assert get_remote_address(request) == "10.0.0.1"


def test_get_remote_address_uses_forwarded_header_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With trusted-proxy mode enabled, use the first forwarded address."""
    monkeypatch.setenv("XP_TRUST_PROXY_HEADERS", "1")
    request = _request(
        client_host="10.0.0.1", forwarded_for="203.0.113.9, 198.51.100.1"
    )
    assert get_remote_address(request) == "203.0.113.9"


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("1/second", (1, 1)),
        ("10/minutes", (10, 60)),
        ("3/hour", (3, 3600)),
        ("2/days", (2, 86_400)),
    ],
)
def test_parse_rule_valid(rule: str, expected: tuple[int, int]) -> None:
    """Valid rules should parse into count and period in seconds."""
    assert _parse_rule(rule) == expected


@pytest.mark.parametrize(
    "rule",
    ["", "abc", "0/minute", "-1/day", "2", "a/minute", "2/week"],
)
def test_parse_rule_invalid(rule: str) -> None:
    """Invalid rules should raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Invalid"):
        _parse_rule(rule)


def test_check_rate_limit_allows_then_blocks() -> None:
    """Rate limiter should block requests after count is reached in a window."""
    limiter = RateLimiter()
    request = _request()

    assert limiter.check_rate_limit(request, "2/second")
    assert limiter.check_rate_limit(request, "2/second")
    assert not limiter.check_rate_limit(request, "2/second")


def test_sweep_stale_keys_removes_old_entries() -> None:
    """Global sweep should remove keys that are stale for the max tracked window."""
    limiter = RateLimiter()
    limiter._storage["stale:/a"] = [1.0]
    limiter._storage["fresh:/a"] = [150_000.0]

    limiter._sweep_stale_keys(200_000.0)

    assert "stale:/a" not in limiter._storage
    assert "fresh:/a" in limiter._storage
