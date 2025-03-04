"""Download paper recommendations for PeerRead papers from the Semantic Scholar API.

The input is the output of the `paper.semantic_scholar.info` script, where we have the
S2 information for the paper. We need this for the paperId, which the recommendation
endpoint uses as input.

The recommendations API offers two pools to get recommendations from: "recent" and
"all-cs". For each paper, we query both pools and return the union of papers.

The output is two files:
- papers_with_recommendations.json: full data - each paper with its list of
  recommendations
- papers_recommended.json: unique recommended papers with the set of titles for the
  papers that led to them. The titles come from PeerRead, not S2.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import backoff
import dotenv
import typer
from aiolimiter import AsyncLimiter

from paper import semantic_scholar as s2
from paper.semantic_scholar.model import (
    Paper,
    PaperRecommended,
    PaperWithRecommendations,
    PeerReadPaperWithS2,
)
from paper.util import (
    arun_safe,
    ensure_envvar,
    get_params,
    progress,
    render_params,
)
from paper.util.cli import die
from paper.util.serde import load_data, save_data

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
S2_RECOMMENDATIONS_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)

MAX_CONCURRENT_REQUESTS = 1
REQUESTS_PER_SECOND = 1


def _get_limiter(
    max_concurrent_requests: int = 1,
    requests_per_second: float = 1,
    use_semaphore: bool | None = None,
) -> asyncio.Semaphore | AsyncLimiter:
    """Create some form of requests limiter based on the `USE_SEMAPHORE` env var.

    Args:
        max_concurrent_requests: When using a semaphore, the maximum number of requests
            (async tasks) that can execute simultaneously. The rest will wait until
            there's room available.
        requests_per_second: Number of requests per second that can be made.
        use_semaphore: Which one to use. If None, will check the USE_SEMAPHORE env var.
            If USE_SEMAPHORE is 1, use a Semaphore. Otherwise, use a rate limiter.

    Use Semaphore with small batches when you're not too worried about the rate limit,
    or Rate Limiiter when you want something more reliable.
    """
    if use_semaphore is None:
        use_semaphore = os.environ.get("USE_SEMAPHORE", "1") == "1"

    if use_semaphore:
        return asyncio.Semaphore(max_concurrent_requests)
    return AsyncLimiter(requests_per_second, 1)


LIMITER = _get_limiter(MAX_CONCURRENT_REQUESTS, REQUESTS_PER_SECOND)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the JSON file containing papers with S2 data and paper_id"
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Path to output directory to save the downloaded papers"),
    ],
    fields: Annotated[
        str,
        typer.Option(help="Comma-separated list of fields to retrieve"),
    ] = ",".join((
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
    )),
    limit_papers: Annotated[
        int | None,
        typer.Option(help="Number of papers to download recommendations from"),
    ] = None,
    limit_recommendations: Annotated[
        int,
        typer.Option(help="Number of recommendations per paper.", max=500),
    ] = 30,
) -> None:
    """Download recommended papers for each paper in the PeerRead dataset.

    Synchronous wrapper around `download_paper_recomendation`. Go there for details.
    """
    dotenv.load_dotenv()

    arun_safe(
        download_paper_recomendation,
        input_file,
        fields,
        output_dir,
        limit_papers,
        limit_recommendations,
    )


async def download_paper_recomendation(
    input_file: Path,
    fields_str: str,
    output_dir: Path,
    limit_papers: int | None,
    limit_recommendations: int,
) -> None:
    """Download recommended papers for each paper in the PeerRead dataset.

    We query the first `limit_papers`, if not None. For each paper, we get
    `limit_recommendations`.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required, as payloads larger than 10 MB will generate
    errors.
    """
    params = get_params()
    print(render_params(params))

    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    if not fields:
        die("No valid --fields. It should be a comma-separated strings of field names.")

    if limit_papers is not None and limit_papers <= 0:
        die(f"Paper limit should be non-negative. Got {limit_papers}.")

    if not (1 <= limit_recommendations <= 500):
        die(
            "Paper recommendations limit should be between 1 and 500. Got"
            f" '{limit_recommendations}'."
        )

    papers = load_data(input_file, PeerReadPaperWithS2)[:limit_papers]

    papers_with_recommendations = await _fetch_recommendations(
        api_key, papers, fields, limit_recommendations
    )
    papers_unique = _merge_papers(papers_with_recommendations)
    papers_unique_valid = [
        paper for paper in papers_unique if paper.abstract and paper.year
    ]

    print(
        "Total papers:",
        sum(len(paper.recommendations) for paper in papers_with_recommendations),
    )
    print("Unique papers:", len(papers_unique))
    print("Unique valid papers (non-empty abstract):", len(papers_unique_valid))

    save_data(
        output_dir / "papers_with_recommendations.json", papers_with_recommendations
    )
    save_data(output_dir / "papers_recommended.json", papers_unique_valid)
    save_data(output_dir / "params.json", params)


async def _fetch_recommendations(
    api_key: str,
    papers: Sequence[PeerReadPaperWithS2],
    fields: Sequence[str],
    limit_recommendations: int,
) -> list[PaperWithRecommendations]:
    """Fetch recommendations from each paper.

    Args:
        api_key: Semantic Scholar API key.
        papers: Paper to get recommendations for.
        fields: Fields to obtain from the API.
        from_: Pool to get the recommendations from.
        limit_recommendations: Maximum number of recommendations to get per paper.

    Returns:
        List of papers with associated recommendations.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT), headers={"x-api-key": api_key}
    ) as session:
        tasks = [
            _fetch_paper_recommendations(
                session, paper.s2, fields, limit_recommendations
            )
            for paper in papers
        ]
        task_results = await progress.gather(tasks, desc="Querying papers")
        return [
            PaperWithRecommendations(main_paper=paper, recommendations=result)
            for paper, result in zip(papers, task_results)
        ]


def _merge_papers(papers: Iterable[PaperWithRecommendations]) -> list[PaperRecommended]:
    """Get unique papers across all recommendations, including data from the inputs.

    It's possible that different papers get the same recommendations, so we're removing
    duplicates. The unique papers also include the title of the papers that recommended
    them. Note that this title comes from the PeerRead dataset, no S2, as they can sometimes
    differ.

    Args:
        papers: Main papers with their recommendations.

    Returns:
        List of the unique S2 papers with the name of the papers that led to them.
    """
    paper_idx: dict[str, Paper] = {}
    paper_sources_peer: defaultdict[str, set[str]] = defaultdict(set)
    paper_sources_s2: defaultdict[str, set[str]] = defaultdict(set)

    for main_paper in papers:
        for paper in main_paper.recommendations:
            if paper.paper_id not in paper_idx:
                paper_idx[paper.paper_id] = paper

            paper_sources_peer[paper.paper_id].add(main_paper.main_paper.title)
            if s2_title := main_paper.main_paper.s2.title:
                paper_sources_s2[paper.paper_id].add(s2_title)

    return [
        PaperRecommended(
            paper_id=paper.paper_id,
            corpus_id=paper.corpus_id,
            url=paper.url,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            abstract=paper.abstract,
            reference_count=paper.reference_count,
            citation_count=paper.citation_count,
            influential_citation_count=paper.influential_citation_count,
            tldr=paper.tldr,
            sources_peer=sorted(paper_sources_peer[paper.paper_id]),
            sources_s2=sorted(paper_sources_s2[paper.paper_id]),
        )
        for paper in paper_idx.values()
    ]


async def _fetch_paper_recommendations(
    session: aiohttp.ClientSession,
    paper: s2.PaperFromPeerRead,
    fields: Iterable[str],
    limit_recommendations: int,
) -> list[Paper]:
    """Fetch paper recommendations from a paper in all VALID_FROM pools.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        paper: S2 paper to be queried through its paperId.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        limit_recommendations: Maximum number of recommendations per paper.
            Must be <= 500.

    Returns:
        List of S2 recommended papers. If there was an error, prints it and returns an
        empty list.
    """
    results = await _fetch_paper_recommendations_from(
        session, paper, fields, limit_recommendations
    )
    return _deduplicate_papers(results)


def _deduplicate_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Remove duplicate papers by paper_id."""
    seen: set[str] = set()
    output: list[Paper] = []

    for paper in papers:
        if paper.paper_id not in seen:
            seen.add(paper.paper_id)
            output.append(paper)

    return output


async def _fetch_paper_recommendations_from(
    session: aiohttp.ClientSession,
    paper: s2.PaperFromPeerRead,
    fields: Iterable[str],
    limit_recommendations: int,
) -> list[Paper]:
    """Fetch paper recommendations for a paper. Only returns data from `fields`.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        paper: S2 paper to be queried through its paperId.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        from_: Pool of papers to recommend from: "recent" or "all-cs".
        limit_recommendations: Maximum number of recommendations per paper.
            Must be <= 500.

    Returns:
        List of S2 recommended papers. If there was an error, prints it and returns an
        empty list.
    """
    params = {
        "from": "all-cs",
        "fields": ",".join(fields),
        "limit": limit_recommendations,
    }
    url = f"{S2_RECOMMENDATIONS_BASE_URL}/{paper.paper_id}"

    try:
        async with LIMITER:
            result = await _fetch_with_retries(session, params=params, url=url)

        if error := result.get("error"):
            print(f"Paper '{paper.title}' failed with error: {error}")
            return []

        if data := result.get("recommendedPapers"):
            return [Paper.model_validate(paper) for paper in data]

    except Exception as e:
        print(f"Paper '{paper.title}' failed after {MAX_RETRIES} tries:")
        print("Last error:", e)

    return []


@backoff.on_exception(
    backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=MAX_RETRIES
)
async def _fetch_with_retries(
    session: aiohttp.ClientSession, *, params: dict[str, Any], url: str
) -> dict[str, Any]:
    """Execute an API request with automatic retrying on HTTP errors and timeouts.

    Uses exponential backoff and jitter with set number of tries.

    Raises:
        aiohttp.ContentTypeError: If the response is not valid JSON.
        aiohttp.ClientError: If the there are any other problems with the request. E.g.
            the server returns a 400 code or higher, or any other aiottp client error.
        asyncio.TimeoutError: If the request runs out time (see `REQUEST_TIMEOUT`).
    """
    async with session.get(url, params=params) as response:
        # 400 and 404 carry error messages as JSON payload, so we handle them manually.
        if response.status > 400 and response.status != 404:
            response.raise_for_status()

        return await response.json()


if __name__ == "__main__":
    app()
