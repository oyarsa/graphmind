"""Tests for keeping Semantic Scholar request volume down.

The S2 key allows roughly one request per second, so every wasted request (and
every retry) slows evaluations down. These tests cover the parts of the pipeline
that avoid unnecessary requests.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self, cast, override
from unittest.mock import AsyncMock, patch

import aiohttp
import numpy as np
import pytest

from paper import peerread as pr
from paper import semantic_scholar as s2
from paper.embedding import Matrix, TextEncoder, Vector
from paper.semantic_scholar import info
from paper.single_paper.paper_retrieval import (
    fetch_s2_recommendations,
    get_top_k_titles_async,
)
from paper.single_paper.s2_store import S2Store
from paper.util.rate_limiter import AsyncLimiter, get_limiter


class _LengthEncoder(TextEncoder):
    """Encodes text by its length, so tests don't need a real model."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @override
    async def encode(self, text: str) -> Vector:
        return np.array([1.0, float(len(text))], dtype=np.float32)

    @override
    async def encode_multi(self, texts: Sequence[str]) -> Matrix:
        return np.stack([await self.encode(t) for t in texts])

    @override
    async def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> Matrix:
        self.batches.append(list(texts))
        return await self.encode_multi(texts)


def _paper(ref_titles: Sequence[str], s2_paper_id: str | None = None) -> pr.Paper:
    return pr.Paper(
        title="Main paper title",
        abstract="Abstract.",
        reviews=[],
        authors=[],
        sections=[],
        approval=None,
        references=[
            pr.PaperReference(title=t, year=2020, authors=[], contexts=[])
            for t in ref_titles
        ],
        conference="",
        s2_paper_id=s2_paper_id,
    )


@pytest.mark.asyncio
async def test_top_k_titles_skips_placeholders_blanks_and_duplicates() -> None:
    """Placeholder, blank and repeated titles never reach the encoder or S2."""
    encoder = _LengthEncoder()
    paper = _paper([
        "Real paper A",
        pr.UNKNOWN_REFERENCE_TITLE,
        "Real paper B",
        pr.UNKNOWN_REFERENCE_TITLE,
        "   ",
        "Real paper A",
    ])

    titles = await get_top_k_titles_async(encoder, paper, k=10)

    assert sorted(titles) == ["Real paper A", "Real paper B"]
    assert encoder.batches == [["Real paper A", "Real paper B"]]


@pytest.mark.asyncio
async def test_top_k_titles_only_placeholders_returns_empty() -> None:
    """A paper whose references are all placeholders has nothing to look up."""
    encoder = _LengthEncoder()
    paper = _paper([pr.UNKNOWN_REFERENCE_TITLE, pr.UNKNOWN_REFERENCE_TITLE])

    assert await get_top_k_titles_async(encoder, paper, k=10) == []
    assert encoder.batches == []


class _RecordingStore(S2Store):
    """S2Store fake that records which methods were called."""

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.recommendation_ids: list[str] = []

    @override
    async def search(
        self, title: str, fields: Sequence[str], limit: int = 1
    ) -> dict[str, Any] | None:
        self.searches.append(title)
        return {"paperId": "searched-id"}

    @override
    async def paper_by_title(self, title: str) -> s2.PaperFromPeerRead | None:
        raise NotImplementedError

    @override
    async def papers_by_titles(
        self, titles: Sequence[str], fields: Sequence[str]
    ) -> list[s2.PaperFromPeerRead | None]:
        raise NotImplementedError

    @override
    async def recommendations(
        self,
        paper_title: str,
        paper_id: str,
        fields: Sequence[str],
        limit: int,
        from_: str,
        request_timeout: float,
    ) -> list[s2.Paper]:
        self.recommendation_ids.append(paper_id)
        return []


@pytest.mark.asyncio
async def test_recommendations_reuse_known_s2_paper_id() -> None:
    """If the paper already has its S2 ID, no extra title search is made."""
    store = _RecordingStore()

    await fetch_s2_recommendations(_paper([], s2_paper_id="known-id"), 5, store, 10)

    assert store.searches == []
    assert store.recommendation_ids == ["known-id", "known-id"]


@pytest.mark.asyncio
async def test_recommendations_search_when_s2_paper_id_unknown() -> None:
    """Without a known S2 ID, the paper is looked up by title first."""
    store = _RecordingStore()

    await fetch_s2_recommendations(_paper([]), 5, store, 10)

    assert store.searches == ["Main paper title"]
    assert store.recommendation_ids == ["searched-id", "searched-id"]


class _RateLimitedResponse:
    """aiohttp-like response that always returns 429 without Retry-After."""

    status = 429
    headers: dict[str, str] = {}  # noqa: RUF012

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool:
        return False


class _RateLimitedSession:
    """aiohttp-like session whose every request is rate limited."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, *_args: Any, **_kwargs: Any) -> _RateLimitedResponse:
        self.calls += 1
        return _RateLimitedResponse()


@pytest.mark.asyncio
async def test_fetch_paper_data_respects_max_retries_without_final_sleep() -> None:
    """Gives up after `max_retries` attempts and doesn't sleep after the last one."""
    session = _RateLimitedSession()
    sleep = AsyncMock()

    with patch.object(info.asyncio, "sleep", sleep):
        result = await info.fetch_paper_data(
            cast(aiohttp.ClientSession, session),
            "key",
            "Some title",
            ["title"],
            AsyncLimiter(1000, 1),
            max_retries=3,
        )

    assert result is None
    assert session.calls == 3
    assert [c.args[0] for c in sleep.await_args_list] == [
        info.RETRY_DELAY,
        info.RETRY_DELAY * info.BACKOFF_FACTOR,
    ]


def test_get_limiter_supports_fractional_rates() -> None:
    """A rate below 1/s spaces single requests out instead of failing."""
    limiter = get_limiter(requests_per_second=0.8, use_semaphore=False)

    assert isinstance(limiter, AsyncLimiter)
    assert limiter.max_rate == 1
    assert limiter.time_period == 1.25  # Exact in binary floating point
