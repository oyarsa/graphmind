"""Search functionality for abstract paper evaluation using the Semantic Scholar API.

This module provides search capabilities for finding papers based on title and abstract
content, designed for the simplified evaluation pipeline that works with unpublished
papers.

Key features:
- Async batched searches for performance
- Content-based and keyword-based queries
- Result deduplication and ranking
- Rate limiting for S2 API compliance
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import aiohttp
import backoff

from paper import semantic_scholar as s2
from paper.single_paper.paper_retrieval import S2_FIELDS_BASE

if TYPE_CHECKING:
    from paper.util.rate_limiter import Limiter

logger = logging.getLogger(__name__)

# S2 API configuration
S2_SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


@backoff.on_exception(
    backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=MAX_RETRIES
)
async def _search_papers_request(
    session: aiohttp.ClientSession,
    api_key: str,
    query: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Make a single search request to the S2 API."""

    params = {
        "query": query,
        "limit": min(limit, 100),  # S2 API limit
        "offset": offset,
        "fields": ",".join(S2_FIELDS_BASE),
    }

    headers = {"X-API-KEY": api_key}

    async with session.get(
        S2_SEARCH_BASE_URL,
        params=params,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
    ) as response:
        if response.status == 429:
            # Rate limited - backoff will retry
            raise aiohttp.ClientError("Rate limited")
        if response.status != 200:
            logger.warning(
                f"S2 search failed with status {response.status}: {await response.text()}"
            )
            return []

        data = await response.json()
        return data.get("data", [])


async def search_papers_by_content(
    limiter: Limiter,
    api_key: str,
    title: str,
    abstract: str,
    limit: int,
) -> list[s2.Paper]:
    """Search for papers using title and abstract content.

    Args:
        limiter: Rate limiter for API requests.
        api_key: Semantic Scholar API key.
        title: Paper title.
        abstract: Paper abstract.
        limit: Maximum number of papers to return.

    Returns:
        List of S2 Paper objects matching the content
    """
    # Construct search query from title and key abstract terms
    # Truncate abstract to avoid overly long queries
    query = f"{title} {abstract[:500]}"

    async with limiter, aiohttp.ClientSession() as session:
        raw_papers = await _search_papers_request(session, api_key, query, limit)

    # Convert to Paper objects, filtering out papers without abstracts
    papers: list[s2.Paper] = []
    for raw_paper in raw_papers:
        try:
            paper = s2.Paper.model_validate(raw_paper)
            if paper.abstract and paper.title:  # Only include papers with content
                papers.append(paper)
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            continue

    return papers


async def search_papers_by_keywords(
    limiter: Limiter,
    api_key: str,
    keywords: Sequence[str],
    limit: int,
) -> list[s2.Paper]:
    """Search for papers using extracted keywords.

    Args:
        limiter: Rate limiter for API requests.
        api_key: Semantic Scholar API key.
        keywords: List of research keywords/terms
        limit: Maximum number of papers to return per keyword

    Returns:
        List of S2 Paper objects matching the keywords
    """
    if not keywords:
        return []

    # Create search tasks for all keywords
    async def search_keyword(keyword: str) -> list[s2.Paper]:
        async with limiter, aiohttp.ClientSession() as session:
            raw_papers = await _search_papers_request(session, api_key, keyword, limit)

        papers: list[s2.Paper] = []
        for raw_paper in raw_papers:
            try:
                paper = s2.Paper.model_validate(raw_paper)
                if paper.abstract and paper.title:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse paper from keyword '{keyword}': {e}")
                continue
        return papers

    # Execute searches in parallel with rate limiting
    tasks = [search_keyword(keyword) for keyword in keywords[:5]]  # Limit to 5 keywords
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results and filter exceptions
    papers: list[s2.Paper] = []
    for result in results:
        if isinstance(result, list):
            papers.extend(result)
        else:
            logger.warning(f"Keyword search failed: {result}")

    return papers


def combine_search_results(
    content_results: Sequence[s2.Paper],
    keyword_results: Sequence[s2.Paper],
    max_results: int,
) -> list[s2.Paper]:
    """Combine and deduplicate search results.

    Prioritises content-based results over keyword-based results,
    then deduplicates by paper ID.

    Args:
        content_results: Papers from title+abstract search.
        keyword_results: Papers from keyword search.
        max_results: Maximum number of papers to return.

    Returns:
        Deduplicated and ranked list of papers.
    """
    seen_ids: set[str] = set()
    combined: list[s2.Paper] = []

    # Prioritise content results
    for paper in valid_papers(content_results):
        if paper.paper_id not in seen_ids:
            combined.append(paper)
            seen_ids.add(paper.paper_id)
            if len(combined) >= max_results:
                break

    # Add keyword results if we need more
    for paper in valid_papers(keyword_results):
        if len(combined) >= max_results:
            break
        if paper.paper_id not in seen_ids:
            combined.append(paper)
            seen_ids.add(paper.paper_id)

    return combined[:max_results]


def _normalise_title(title: str) -> str:
    """Remove non-alpha characters from title."""
    return "".join(c for c in title if c.isalpha() or c.isspace()).strip()


async def search_related_papers(
    limiter: Limiter,
    api_key: str,
    title: str,
    abstract: str,
    keywords: Sequence[str] | None,
    max_results: int,
) -> list[s2.Paper]:
    """Search for papers related to the given title and abstract.

    Combines content-based search with keyword-based search for comprehensive
    coverage of related work.

    Args:
        limiter: Rate limiter for API requests.
        api_key: Semantic Scholar API key.
        title: Paper title.
        abstract: Paper abstract.
        keywords: Optional list of research keywords.
        max_results: Maximum number of papers to return.
        fields: S2 API fields to retrieve.

    Returns:
        List of related papers ranked by relevance.
    """
    # Run content and keyword searches in parallel
    tasks = [search_papers_by_content(limiter, api_key, title, abstract, max_results)]

    if keywords:
        tasks.append(
            search_papers_by_keywords(
                limiter, api_key, keywords, max_results // len(keywords)
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    content_results = results[0] if isinstance(results[0], list) else []
    keyword_results = (
        results[1] if len(results) > 1 and isinstance(results[1], list) else []
    )

    combined = combine_search_results(content_results, keyword_results, max_results)
    # Remove results with the same title as the main paper
    return [
        p
        for p in combined
        if p.title and _normalise_title(p.title) != _normalise_title(title)
    ]


def valid_papers(papers: Sequence[s2.Paper]) -> Iterable[s2.Paper]:
    """Filter papers to keep only those with non-empty abstracts and titles."""

    def is_valid(s: str | None) -> bool:
        return s is not None and s.strip() != ""

    return (p for p in papers if is_valid(p.abstract) and is_valid(p.title))
