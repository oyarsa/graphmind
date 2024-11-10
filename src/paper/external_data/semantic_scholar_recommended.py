"""Download paper recommendations for ASAP papers from the Semantic Scholar API.

The input is the output of the `paper.external_data.semantic_scholar_info` script, where
we have the S2 information for the paper. We need this for the paperId, which the
recommendation endpoint uses as input.

The output is two files:
- papers_with_recommendations.json: full data - each paper with its list of
  recommendations
- papers_recommended.json: unique recommended papers with the set of titles for the
  papers that led to them. The titles come from ASAP, not S2.
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import aiohttp
import backoff
import dotenv

from paper import progress
from paper.external_data.semantic_scholar_model import (
    ASAPPaperWithS2,
    PaperRecommended,
    PaperWithRecommendations,
)
from paper.external_data.semantic_scholar_model import Paper as S2Paper
from paper.util import (
    HelpOnErrorArgumentParser,
    arun_safe,
    display_params,
    ensure_envvar,
    get_limiter,
    load_data,
    save_data,
)

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
S2_RECOMMENDATIONS_BASE_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
VALID_FROM = ("recent", "all-cs")

MAX_CONCURRENT_REQUESTS = 1
REQUESTS_PER_SECOND = 1
LIMITER = get_limiter(MAX_CONCURRENT_REQUESTS, REQUESTS_PER_SECOND)


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        help="The path to the JSON file containing papers with S2 data and paper_id.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to output directory to save the downloaded papers.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=",".join(
            (
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
            )
        ),
        help="Comma-separated list of fields to retrieve.",
    )
    parser.add_argument(
        "--from",
        dest="from_",
        type=str,
        default="recent",
        help="Pool to recommend papers from",
        choices=VALID_FROM,
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        default=None,
        help="Number of papers to download recommendations from.",
    )
    parser.add_argument(
        "--limit-recommendations",
        type=int,
        default=100,
        help="Number of recommendations per paper. Must be <=500.",
    )
    args = parser.parse_args()

    arun_safe(
        download_paper_recomendation,
        args.input_file,
        args.fields,
        args.from_,
        args.output_dir,
        args.limit_papers,
        args.limit_recommendations,
    )


async def download_paper_recomendation(
    input_file: Path,
    fields_str: str,
    from_: str,
    output_dir: Path,
    limit_papers: int | None,
    limit_recommendations: int,
) -> None:
    """Download recommended papers for each paper in the ASAP dataset.

    We query the first `limit_papers`, if not None. For each paper, we get
    `limit_recommendations`.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required, as payloads larger than 10 MB will generate
    errors.
    """
    print(display_params())

    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    if not fields:
        sys.exit(
            "No valid --fields. It should be a comma-separated strings of field names."
        )

    if limit_papers is not None and limit_papers <= 0:
        sys.exit(f"Paper limit should be non-negative. Got {limit_papers}.")

    if not (1 <= limit_recommendations <= 500):
        sys.exit(
            "Paper recommendations limit should be between 1 and 500. Got"
            f" '{limit_recommendations}'."
        )

    papers = load_data(input_file, ASAPPaperWithS2)[:limit_papers]

    papers_with_recommendations = await _fetch_recommendations(
        api_key, papers, fields, from_, limit_recommendations
    )
    papers_unique = _merge_papers(papers_with_recommendations)

    print(
        "Total papers:",
        sum(len(paper.recommendations) for paper in papers_with_recommendations),
    )
    print("Unique papers:", len(papers_unique))

    save_data(
        output_dir / "papers_with_recommendations.json", papers_with_recommendations
    )
    save_data(output_dir / "papers_recommended.json", papers_unique)


async def _fetch_recommendations(
    api_key: str,
    papers: Sequence[ASAPPaperWithS2],
    fields: Sequence[str],
    from_: str,
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
                session, paper.s2, fields, from_, limit_recommendations
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
    them. Note that this title comes from the ASAP dataset, no S2, as they can sometimes
    differ.

    Args:
        papers: Main papers with their recommendations.

    Returns:
        List of the unique S2 papers with the name of the papers that led to them.
    """
    paper_idx: dict[str, S2Paper] = {}
    paper_sources_asap: defaultdict[str, set[str]] = defaultdict(set)
    paper_sources_s2: defaultdict[str, set[str]] = defaultdict(set)

    for main_paper in papers:
        for paper in main_paper.recommendations:
            if paper.paper_id not in paper_idx:
                paper_idx[paper.paper_id] = paper

            paper_sources_asap[paper.paper_id].add(main_paper.main_paper.title)
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
            sources_asap=sorted(paper_sources_asap[paper.paper_id]),
            sources_s2=sorted(paper_sources_s2[paper.paper_id]),
        )
        for paper in paper_idx.values()
    ]


async def _fetch_paper_recommendations(
    session: aiohttp.ClientSession,
    paper: S2Paper,
    fields: Iterable[str],
    from_: str,
    limit_recommendations: int,
) -> list[S2Paper]:
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
    if from_ not in VALID_FROM:
        raise ValueError(f"Invalid 'from' value. Must be one of: {VALID_FROM}")

    params = {
        "from": from_,
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
            return [S2Paper.model_validate(paper) for paper in data]

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
    main()
