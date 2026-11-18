"""Abstraction over Semantic Scholar data access for the single-paper pipeline.

`S2Store` is the interface the retrieval/pipeline code depends on instead of
talking to the Semantic Scholar API directly. `LiveS2Store` reproduces the exact
calls that used to be made inline. A caching implementation can wrap any
`S2Store` without the pipeline code needing to know the difference.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from pydantic import BaseModel, Field

from paper import (
    semantic_scholar as s2,  # noqa: TC001 (needed at runtime by _CacheData)
)
from paper.semantic_scholar.info import (
    fetch_paper_data,
    fetch_paper_info,
    fetch_papers_from_s2,
)
from paper.semantic_scholar.recommended import fetch_paper_recommendations

if TYPE_CHECKING:
    from paper.util.rate_limiter import Limiter

# Base S2 fields common to all queries
S2_FIELDS_BASE = [
    "paperId",
    "corpusId",
    "url",
    "title",
    "authors",
    "year",
    "publicationDate",
    "abstract",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
]

# Full fields including venue and tldr
S2_FIELDS = [*S2_FIELDS_BASE, "tldr", "venue"]

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
REFERENCE_MAX_RETRIES = 3
"""Attempts per reference lookup. Fewer than the main paper: a missing reference
only slightly degrades the evaluation, so long backoffs aren't worth it."""


def _normalise_title(title: str) -> str:
    """Remove non-alpha characters from title."""
    return "".join(c for c in title if c.isalpha()).casefold()


async def fetch_s2_paper_info(
    api_key: str, title: str, limiter: Limiter
) -> s2.PaperFromPeerRead | None:
    """Fetch paper information from the Semantic Scholar API.

    If the paper retrieved from S2 does not have the same (normalised) title as
    `title`, returns None too. This is because the S2 API will happily return
    something else with a different title if the original was not found.

    Args:
        api_key: Semantic Scholar API key.
        title: Paper title to search for.
        limiter: "Limiter" for the API requests.

    Returns:
        Paper found or None for failed/not found papers.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT)
    ) as session:
        info = await fetch_paper_info(session, api_key, title, S2_FIELDS, limiter)
        if info is None:
            return None
        if _normalise_title(info.title) != _normalise_title(title):
            return None
        return info


class S2Store(ABC):
    """Access to Semantic Scholar paper data, decoupled from how it's fetched."""

    @abstractmethod
    async def search(
        self, title: str, fields: Sequence[str], limit: int = 1
    ) -> dict[str, Any] | None:
        """Raw S2 title search (e.g. for cheap paperId resolution)."""

    @abstractmethod
    async def paper_by_title(self, title: str) -> s2.PaperFromPeerRead | None:
        """Typed, title-validated single-paper lookup."""

    @abstractmethod
    async def papers_by_titles(
        self, titles: Sequence[str], fields: Sequence[str]
    ) -> list[s2.PaperFromPeerRead | None]:
        """Batch title lookup."""

    @abstractmethod
    async def recommendations(
        self,
        paper_title: str,
        paper_id: str,
        fields: Sequence[str],
        limit: int,
        from_: str,
        request_timeout: float,
    ) -> list[s2.Paper]:
        """S2 recommendations for a paper, from a given recommendation pool."""


class LiveS2Store(S2Store):
    """Talks to the real Semantic Scholar API. No caching."""

    def __init__(self, api_key: str, limiter: Limiter) -> None:
        self._api_key = api_key
        self._limiter = limiter

    async def search(
        self, title: str, fields: Sequence[str], limit: int = 1
    ) -> dict[str, Any] | None:
        """Raw S2 title search (e.g. for cheap paperId resolution)."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT)
        ) as session:
            return await fetch_paper_data(
                session, self._api_key, title, fields, self._limiter, limit
            )

    async def paper_by_title(self, title: str) -> s2.PaperFromPeerRead | None:
        """Typed, title-validated single-paper lookup."""
        return await fetch_s2_paper_info(self._api_key, title, self._limiter)

    async def papers_by_titles(
        self, titles: Sequence[str], fields: Sequence[str]
    ) -> list[s2.PaperFromPeerRead | None]:
        """Batch title lookup."""
        return await fetch_papers_from_s2(
            self._api_key,
            titles,
            fields,
            limiter=self._limiter,
            max_retries=REFERENCE_MAX_RETRIES,
        )

    async def recommendations(
        self,
        paper_title: str,
        paper_id: str,
        fields: Sequence[str],
        limit: int,
        from_: str,
        request_timeout: float,
    ) -> list[s2.Paper]:
        """S2 recommendations for a paper, from a given recommendation pool."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(request_timeout),
            headers={"x-api-key": self._api_key},
        ) as session:
            return await fetch_paper_recommendations(
                session,
                paper_title,
                paper_id,
                fields,
                limit,
                self._limiter,
                from_=from_,
            )


def _recommendations_key(paper_id: str, from_: str, limit: int) -> str:
    """Build a cache key for a recommendations lookup."""
    return f"{paper_id}|{from_}|{limit}"


class _CacheData(BaseModel):
    """On-disk shape of a `CachingS2Store` cache file."""

    search: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    paper_by_title: dict[str, s2.PaperFromPeerRead | None] = Field(default_factory=dict)
    papers_by_titles: dict[str, s2.PaperFromPeerRead | None] = Field(
        default_factory=dict
    )
    recommendations: dict[str, list[s2.Paper]] = Field(default_factory=dict)


class CachingS2Store(S2Store):
    """Memoises another `S2Store`'s results to an in-memory cache, persisted to disk.

    Concurrent callers asking for the same not-yet-cached key share a single
    in-flight fetch rather than each firing their own request. See `_get_or_fetch`.
    """

    def __init__(self, inner: S2Store, cache_path: Path | None = None) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._lock = asyncio.Lock()

        data = (
            _CacheData.model_validate_json(cache_path.read_bytes())
            if cache_path is not None and cache_path.exists()
            else _CacheData()
        )
        self._search_cache = data.search
        self._paper_by_title_cache = data.paper_by_title
        self._papers_by_titles_cache = data.papers_by_titles
        self._recommendations_cache = data.recommendations

        self._search_pending: dict[str, asyncio.Task[dict[str, Any] | None]] = {}
        self._paper_by_title_pending: dict[
            str, asyncio.Task[s2.PaperFromPeerRead | None]
        ] = {}
        self._papers_by_titles_pending: dict[
            str, asyncio.Task[s2.PaperFromPeerRead | None]
        ] = {}
        self._recommendations_pending: dict[str, asyncio.Task[list[s2.Paper]]] = {}

    async def _get_or_fetch[T](
        self,
        cache: dict[str, T],
        pending: dict[str, asyncio.Task[T]],
        key: str,
        fetch: Callable[[], Awaitable[T]],
    ) -> T:
        """Return `cache[key]`, fetching it at most once even under concurrent callers.

        The lock only guards the synchronous check-and-launch step (dict lookups
        plus scheduling a task), not the fetch itself, so it doesn't serialise
        network calls — only the bookkeeping around starting them.
        """
        async with self._lock:
            if key in cache:
                return cache[key]
            if key not in pending:
                pending[key] = asyncio.ensure_future(fetch())
            task = pending[key]
        result = await task
        cache[key] = result
        pending.pop(key, None)
        return result

    async def search(
        self, title: str, fields: Sequence[str], limit: int = 1
    ) -> dict[str, Any] | None:
        """Raw S2 title search (e.g. for cheap paperId resolution)."""
        key = _normalise_title(title)
        return await self._get_or_fetch(
            self._search_cache,
            self._search_pending,
            key,
            lambda: self._inner.search(title, fields, limit),
        )

    async def paper_by_title(self, title: str) -> s2.PaperFromPeerRead | None:
        """Typed, title-validated single-paper lookup."""
        key = _normalise_title(title)
        return await self._get_or_fetch(
            self._paper_by_title_cache,
            self._paper_by_title_pending,
            key,
            lambda: self._inner.paper_by_title(title),
        )

    async def papers_by_titles(
        self, titles: Sequence[str], fields: Sequence[str]
    ) -> list[s2.PaperFromPeerRead | None]:
        """Batch title lookup, decomposed into per-title cache entries.

        Each title is memoised individually (not as a whole-batch entry) so
        overlapping reference lists across different papers reuse hits instead of
        only deduplicating identical batches.
        """
        return list(
            await asyncio.gather(
                *(self._paper_by_titles_one(title, fields) for title in titles)
            )
        )

    async def _paper_by_titles_one(
        self, title: str, fields: Sequence[str]
    ) -> s2.PaperFromPeerRead | None:
        key = _normalise_title(title)

        async def fetch() -> s2.PaperFromPeerRead | None:
            (result,) = await self._inner.papers_by_titles([title], fields)
            return result

        return await self._get_or_fetch(
            self._papers_by_titles_cache, self._papers_by_titles_pending, key, fetch
        )

    async def recommendations(
        self,
        paper_title: str,
        paper_id: str,
        fields: Sequence[str],
        limit: int,
        from_: str,
        request_timeout: float,
    ) -> list[s2.Paper]:
        """S2 recommendations for a paper, from a given recommendation pool."""
        key = _recommendations_key(paper_id, from_, limit)
        return await self._get_or_fetch(
            self._recommendations_cache,
            self._recommendations_pending,
            key,
            lambda: self._inner.recommendations(
                paper_title, paper_id, fields, limit, from_, request_timeout
            ),
        )

    def flush(self) -> None:
        """Write the current in-memory cache state to `cache_path`.

        No-op if no `cache_path` was given — the cache then only lives for
        the process's lifetime.
        """
        if self._cache_path is None:
            return
        data = _CacheData(
            search=self._search_cache,
            paper_by_title=self._paper_by_title_cache,
            papers_by_titles=self._papers_by_titles_cache,
            recommendations=self._recommendations_cache,
        )
        self._cache_path.write_bytes(data.model_dump_json(indent=2).encode())
