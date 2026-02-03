"""Paper retrieval functionality for fetching papers from arXiv and Semantic Scholar.

This module handles retrieving paper data from external sources including:
- arXiv paper search and metadata retrieval
- Semantic Scholar paper information and recommendations
- Reference enhancement with S2 data
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import aiohttp
import arxiv  # type: ignore

from paper import embedding as emb
from paper import peerread as pr
from paper import semantic_scholar as s2
from paper.orc.arxiv_api import (
    ArxivResult,
    arxiv_from_id,
    arxiv_id_from_url,
    arxiv_search,
    check_latex_availability,
    similar_titles,
)
from paper.orc.download import parse_arxiv_latex
from paper.orc.latex_parser import SentenceSplitter
from paper.semantic_scholar.info import (
    fetch_paper_data,
    fetch_paper_info,
    fetch_papers_from_s2,
)
from paper.semantic_scholar.recommended import fetch_paper_recommendations
from paper.util import atimer

if TYPE_CHECKING:
    from paper.gpt.openai_encoder import OpenAIEncoder
    from paper.util.rate_limiter import Limiter

logger = logging.getLogger(__name__)

type ProgressCallback = Callable[[str], Awaitable[None]]


class ArxivRateLimitError(Exception):
    """Raised when arXiv API returns a rate limit error (HTTP 429)."""


# Base S2 fields common to all queries
S2_FIELDS_BASE = [
    "paperId",
    "corpusId",
    "url",
    "title",
    "authors",
    "year",
    "abstract",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
]

# Full fields including venue and tldr
S2_FIELDS = [*S2_FIELDS_BASE, "tldr", "venue"]

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request


async def get_paper_from_title(title: str, limiter: Limiter, api_key: str) -> pr.Paper:
    """Get a single processed Paper from a title using Semantic Scholar and arXiv.

    Args:
        title: Paper title to search for.
        limiter: "Limiter" for the requests.
        api_key: Semantic Scholar API key.

    Returns:
        Paper object with S2 metadata and parsed arXiv sections/references.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    # Run S2 and arXiv queries in parallel
    s2_paper, arxiv_result = await asyncio.gather(
        atimer(fetch_s2_paper_info(api_key, title, limiter), 3),
        atimer(get_arxiv_from_title(title), 3),
    )

    if not s2_paper:
        raise ValueError(f"Paper not found on Semantic Scholar: {title}")

    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {title}")
    logger.debug("arXiv result: %s", arxiv_result)

    # Parse arXiv LaTeX
    sections, references = await atimer(
        asyncio.to_thread(parse_arxiv_latex, arxiv_result, SentenceSplitter()), 3
    )

    return pr.Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
        arxiv_id=arxiv_result.id,
        arxiv_summary=arxiv_result.summary,
    )


async def get_paper_from_arxiv_id(
    arxiv_id: str,
    limiter: Limiter,
    api_key: str,
    *,
    callback: ProgressCallback | None = None,
) -> pr.Paper:
    """Get a single processed Paper from the arXiv ID using Semantic Scholar and arXiv.

    Args:
        arxiv_id: ID of the paper on arXiv.
        limiter: "Limiter" for the requests.
        api_key: Semantic Scholar API key.
        callback: Optional callback function to call with phase names after completion.

    Returns:
        Paper object with S2 metadata and parsed arXiv sections/references.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    # First get arXiv result to get the title for S2 lookup
    if callback:
        await callback("Fetching arXiv data")

    arxiv_result = await atimer(get_arxiv_from_id(arxiv_id), 3)
    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {arxiv_id}")
    logger.debug("arXiv result: %s - %s", arxiv_id, arxiv_result.arxiv_title)

    # Run S2 lookup and LaTeX parsing in parallel
    if callback:
        await callback("Fetching Semantic Scholar data and parsing arXiv paper")

    s2_paper, (sections, references) = await asyncio.gather(
        atimer(
            fetch_s2_paper_info(api_key, arxiv_result.arxiv_title, limiter=limiter), 3
        ),
        atimer(
            asyncio.to_thread(parse_arxiv_latex, arxiv_result, SentenceSplitter()), 3
        ),
    )

    if not s2_paper:
        logger.warning("Paper not found on S2 API: %s", arxiv_result.arxiv_title)
        raise ValueError(
            "Paper not found on Semantic Scholar. Try another paper, or try the"
            " Abstract tab."
        )

    return pr.Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
        arxiv_id=arxiv_id,
        arxiv_summary=arxiv_result.summary,
    )


async def get_arxiv_from_title(openreview_title: str) -> ArxivResult | None:
    """Get ArxivResult for a single OpenReview paper title if it exists on arXiv."""
    client = arxiv.Client()

    query = f'ti:"{openreview_title}"'

    try:
        result = next(await asyncio.to_thread(arxiv_search, client, query, 1))
        if similar_titles(openreview_title, result.title):
            return ArxivResult(
                id=arxiv_id_from_url(result.entry_id),
                openreview_title=openreview_title,
                arxiv_title=result.title,
                summary=result.summary,
            )
    except Exception as e:
        logger.warning(f"Error searching for '{openreview_title}' on arXiv: {e}")

    return None


async def get_arxiv_from_id(arxiv_id: str) -> ArxivResult | None:
    """Get ArxivResult for a single OpenReview paper title if it exists on arXiv."""
    client = arxiv.Client()

    try:
        result = next(await asyncio.to_thread(arxiv_from_id, client, arxiv_id))
        return ArxivResult(
            id=arxiv_id,
            openreview_title=result.title,
            arxiv_title=result.title,
            summary=result.summary,
        )
    except Exception as e:
        logger.warning(f"Error searching for '{arxiv_id}' on arXiv: {e}")

    return None


async def fetch_s2_paper_info(
    api_key: str, title: str, limiter: Limiter
) -> s2.PaperFromPeerRead | None:
    """Fetch paper information from the Semantic Scholar API.

    If the paper retrieved from S2 does not have the same (normalised) title as `title`,
    returns None too. This is because the S2 API will happily return somethines else
    with a different title if the original was not found.

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
            logger.debug("Invalid title. Original: '%s'. Got: '%s'", title, info.title)
            return None
        return info


def _normalise_title(title: str) -> str:
    """Remove non-alpha characters from title."""
    return "".join(c for c in title if c.isalpha()).casefold()


async def search_arxiv_papers(
    query: str, max_results: int = 10
) -> list[arxiv.Result] | None:
    """Search arXiv for papers matching the query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        None: API error occurred.
        Empty list: successful query but no results found.
        Non-empty list: successful query with results.

    Raises:
        ArxivRateLimitError: If arXiv returns HTTP 429 (rate limited).
    """
    client = arxiv.Client()

    try:
        return list(await asyncio.to_thread(arxiv_search, client, query, max_results))
    except arxiv.HTTPError as e:
        if e.status == 429:
            logger.warning("arXiv rate limit hit: %s", e)
            raise ArxivRateLimitError("arXiv rate limit exceeded") from e
        logger.warning("HTTP error searching arXiv: %s", e)
    except Exception as e:
        logger.warning("Error searching arXiv: %s", e)

    return None


async def search_arxiv_papers_filtered(
    query: str,
    max_results: int,
    *,
    check_latex: bool,
) -> list[arxiv.Result] | None:
    """Search arXiv for papers with optional filtering by LaTeX availability.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        check_latex: Whether to filter by LaTeX availability on arXiv.

    Returns:
        None: API error occurred.
        Empty list: successful query but no results found.
        Non-empty list: successful query with filtered results.
    """

    # First get the raw arXiv results
    # Get more to account for filtering
    search_results = await search_arxiv_papers(query, max_results * 2)
    if search_results is None:
        return None

    if not check_latex:
        return search_results[:max_results]

    filter_tasks = [_filter_paper_result(result) for result in search_results]
    filter_results = await atimer(
        asyncio.gather(*filter_tasks, return_exceptions=True), 2
    )

    filtered_papers: list[arxiv.Result] = []
    for paper, passes_filter in zip(search_results, filter_results):
        match passes_filter:
            case passes_filter if passes_filter:
                filtered_papers.append(paper)
            case BaseException() as e:
                logger.debug(f"Error filtering paper '{passes_filter}': {e}")
            case _:
                pass

    return filtered_papers[:max_results]


async def _filter_paper_result(result: arxiv.Result) -> bool:
    """Check if paper has its LaTeX code on arXiv.

    Returns:
        True if the LaTex code exists, false otherwise.
    """
    try:
        return await check_latex_availability(arxiv_id_from_url(result.entry_id))
    except Exception as e:
        logger.debug(f"Error in filter task: {e}")
        return False


async def get_top_k_titles_async(
    encoder: OpenAIEncoder, paper: pr.Paper, k: int
) -> list[str]:
    """Get top `k` reference titles from `paper` using async encoder.

    References are sorted by cosine similarity between the reference and main paper
    titles.
    """
    if not paper.references:
        return []

    ref_titles = [r.title for r in paper.references]

    references_emb = await encoder.batch_encode(ref_titles)
    title_emb = await encoder.encode(paper.title)
    sim = emb.similarities(title_emb, references_emb)

    return [ref_titles[idx] for idx in emb.top_k_indices(sim, k)]


async def enhance_with_s2_references(
    paper: pr.Paper, top_k: int, encoder: OpenAIEncoder, api_key: str, limiter: Limiter
) -> s2.PaperWithS2Refs:
    """Enhance paper with S2 reference information for top-k similar references."""
    # Get top-k reference titles by semantic similarity
    top_ref_titles = await get_top_k_titles_async(encoder, paper, top_k)

    if not top_ref_titles:
        logger.warning("No references found for paper: %s", paper.title)
        # Return paper with empty S2 references
        return s2.PaperWithS2Refs.from_peer(paper, [])

    # Fetch S2 data for the top references
    s2_results = await fetch_papers_from_s2(
        api_key, top_ref_titles, [*S2_FIELDS_BASE, "tldr"], limiter=limiter
    )

    # Create S2Reference objects by matching with original references
    s2_papers_from_query = {
        s2.clean_title(paper.title_peer): paper for paper in s2_results if paper
    }

    s2_references = [
        s2.S2Reference.from_(
            s2_paper, contexts=ref.contexts, citation_key=ref.citation_key
        )
        for ref in paper.references
        if (s2_paper := s2_papers_from_query.get(s2.clean_title(ref.title)))
    ]

    logger.debug(f"{len(paper.references)=}, {len(s2_results)=}, {len(s2_references)=}")
    return s2.PaperWithS2Refs.from_peer(paper, s2_references)


async def fetch_s2_recommendations(
    paper: pr.Paper,
    num_recommendations: int,
    api_key: str,
    limiter: Limiter,
    request_timeout: float,
) -> list[s2.Paper]:
    """Fetch recommended papers from S2 API for the given paper."""

    # First, we need to get the S2 paper ID by searching for this paper
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
    ) as session:
        data = await fetch_paper_data(
            session, api_key, paper.title, ["paperId"], limiter
        )

    paper_id = (data or {}).get("paperId")
    if not paper_id:
        logger.warning(
            "Paper '%s' not found on S2 - cannot fetch recommendations", paper.title
        )
        return []

    # Query both pools and union results to get recent papers + older CS papers
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(request_timeout), headers={"x-api-key": api_key}
    ) as session:
        recent_task = fetch_paper_recommendations(
            session,
            paper.title,
            paper_id,
            S2_FIELDS_BASE,
            num_recommendations,
            limiter,
            from_="recent",
        )
        all_cs_task = fetch_paper_recommendations(
            session,
            paper.title,
            paper_id,
            S2_FIELDS_BASE,
            num_recommendations,
            limiter,
            from_="all-cs",
        )
        recent_papers, all_cs_papers = await asyncio.gather(recent_task, all_cs_task)

    # Deduplicate by paper_id, preferring recent pool order
    seen: set[str] = set()
    combined: list[s2.Paper] = []
    for rec_paper in recent_papers + all_cs_papers:
        if rec_paper.paper_id not in seen:
            seen.add(rec_paper.paper_id)
            combined.append(rec_paper)
    return combined
