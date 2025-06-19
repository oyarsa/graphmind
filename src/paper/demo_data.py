"""Extract only the data relevant for the demonstration tool.

Includes only entries with a valid rationale, accurate novelty label and non-empty
related papers.

Input format: `gpt.PromptResult[gpt.GraphResult]`.
Output format: `DemoPaper` (in this file).
"""

import asyncio
import io
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import aiohttp
import typer
from pydantic import Field

from paper import gpt
from paper import peerread as pr
from paper.gpt.evaluate_paper import GPTStructured
from paper.gpt.model import PaperTerms, is_rationale_valid
from paper.semantic_scholar.info import (
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    fetch_paper_data,
)
from paper.semantic_scholar.recommended import Limiter
from paper.types import Immutable, PaperProtocol
from paper.util import ensure_envvar, progress, sample, setup_logging
from paper.util.serde import Compress, load_data, replace_fields, save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


def _get_year(paper: PaperProtocol) -> int | None:
    """Get year from paper. If the year is None, try to parse it from the conference."""
    if paper.year is not None:
        return paper.year

    conf_year = paper.conference[-4:]
    try:
        return int(conf_year)
    except ValueError:
        return None


def _clean_conference(conference: str) -> str:
    """Get conference name, excluding the year if present."""
    if len(conference) >= 4 and conference[-4:].isdigit():
        return conference[:-4]
    else:
        return conference


class DemoPaper(Immutable):
    """Paper information."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    year: Annotated[int | None, Field(description="Paper publication year")] = None
    id: Annotated[
        str,
        Field(description="Unique ID for the paper based on the title and abstract."),
    ]

    y_true: Annotated[int, Field(description="Human annotation")]
    y_pred: Annotated[int, Field(description="Model prediction")]
    rationale_true: Annotated[str, Field(description="Human rationale annotation")]
    rationale_pred: Annotated[
        str, Field(description="Model rationale for the prediction")
    ]
    structured_evaluation: Annotated[
        GPTStructured | None,
        Field(description="Structured evaluation breakdown if available"),
    ] = None
    arxiv_id: Annotated[str | None, Field(description="ID of the paper on arXiv")] = (
        None
    )


class DemoData(Immutable):
    """Entry for the data used by the demonstration tool."""

    graph: gpt.Graph
    related: Sequence[gpt.PaperRelatedSummarised]
    paper: DemoPaper

    terms: PaperTerms | None = None
    background: str | None = None
    target: str | None = None


async def fetch_arxiv_id_from_s2(
    session: aiohttp.ClientSession,
    api_key: str,
    title: str,
    limiter: Limiter,
) -> str | None:
    """Fetch arXiv ID from Semantic Scholar API by searching for the paper title.

    Returns the arXiv ID if found in the externalIds field, None otherwise.
    """
    fields = ["externalIds", "title"]
    data = await fetch_paper_data(session, api_key, title, fields, limiter)

    if data is None:
        return None

    external_ids = data.get("externalIds", {})
    if external_ids and "ArXiv" in external_ids:
        return external_ids["ArXiv"]

    return None


async def fetch_arxiv_ids_from_s2(
    titles: Sequence[str], *, desc: str = "Fetching arXiv IDs from S2"
) -> dict[str, str | None]:
    """Fetch arXiv IDs for multiple titles from Semantic Scholar API.

    Returns a mapping from title to arXiv ID (or None if not found).
    """
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
    ) as session:
        limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        tasks = [
            fetch_arxiv_id_from_s2(session, api_key, title, limiter) for title in titles
        ]
        results = await progress.gather(tasks, desc=desc)

        return dict(zip(titles, results))


def fetch_missing_arxiv_ids(
    papers: Sequence[DemoData],
) -> tuple[Sequence[DemoData], int]:
    """Fetch missing arXiv IDs from Semantic Scholar and return updated papers list.

    Returns a tuple of (updated_papers_list, count_of_fetched_ids).
    """
    # Find papers without arXiv IDs
    papers_missing = [p for p in papers if p.paper.arxiv_id is None]
    if not papers_missing:
        return papers, 0

    logger.info(f"Found {len(papers_missing)} papers without arXiv IDs")

    titles = [p.paper.title for p in papers_missing]
    title_to_arxiv_id = asyncio.run(fetch_arxiv_ids_from_s2(titles))

    # Create a new list with updated papers
    updated_papers: list[DemoData] = []
    updated_count = 0

    for paper in papers:
        # Check if this paper needs an arXiv ID update
        if paper.paper.arxiv_id is None and (
            arxiv_id := title_to_arxiv_id.get(paper.paper.title)
        ):
            # Create updated paper with arXiv ID
            updated_papers.append(
                DemoData(
                    graph=paper.graph,
                    related=paper.related,
                    paper=replace_fields(paper.paper, arxiv_id=arxiv_id),
                    terms=paper.terms,
                    background=paper.background,
                    target=paper.target,
                )
            )
            updated_count += 1
        else:
            # Keep paper unchanged
            updated_papers.append(paper)

    logger.info(f"Updated {updated_count} papers with arXiv IDs")
    return updated_papers, updated_count


def display_arxiv_summary(
    total_papers: int, existing_count: int, updated_count: int, fetch_enabled: bool
) -> str:
    """Print summary statistics for arXiv ID fetching."""
    out = io.StringIO()

    print("=== arXiv ID Summary ===", file=out)
    print(f"Total papers: {total_papers}", file=out)
    print(f"Already had arXiv ID: {existing_count}", file=out)
    print(f"Missing arXiv ID: {total_papers - existing_count}", file=out)

    if not fetch_enabled:
        return out.getvalue()

    print("Using Semantic Scholar API:", file=out)
    print(f"Successfully fetched: {updated_count}", file=out)
    print(f"Still missing: {total_papers - existing_count - updated_count}", file=out)

    missing_count = total_papers - existing_count
    if missing_count > 0:
        success_rate = updated_count / missing_count
        print(f"Success rate: {success_rate:.2%}", file=out)

    return out.getvalue()


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Graph evaluation output.")
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSON file."),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", help="How many items to sample. Defaults to all."
        ),
    ] = 10,
    seed: Annotated[int, typer.Option(help="Random seed used for sampling")] = 0,
    fetch_arxiv: Annotated[
        bool,
        typer.Option(help="Fetch missing arXiv IDs from Semantic Scholar."),
    ] = True,
) -> None:
    """Extract only relevant data from the input file."""
    setup_logging()
    rng = random.Random(seed)

    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.GraphResult])
    )

    papers_valid = [
        p
        for p in papers
        if p.paper.y_pred == p.paper.y_true
        and is_rationale_valid(p.rationale_pred)
        and p.related
    ]
    papers_sampled = sample(papers_valid, limit, rng)

    logger.info(f"{len(papers) = }")
    logger.info(f"{len(papers_valid) = }")
    logger.info(f"{len(papers_sampled) = }")

    # Count existing arxiv_ids before conversion
    existing_arxiv_count = sum(
        1 for p in papers_sampled if p.paper.arxiv_id is not None
    )

    papers_converted = [
        DemoData(
            graph=p.graph,
            related=p.related or [],
            paper=DemoPaper(
                title=p.paper.title,
                abstract=p.paper.abstract,
                authors=p.paper.authors,
                sections=p.paper.sections,
                approval=p.paper.approval,
                conference=_clean_conference(p.paper.conference),
                rating=p.paper.rating,
                year=_get_year(p.paper),
                id=p.id,
                y_true=p.paper.y_true,
                y_pred=p.paper.y_pred,
                rationale_true=p.paper.rationale_true,
                rationale_pred=p.rationale_pred,
                structured_evaluation=p.paper.structured_evaluation,
                arxiv_id=p.paper.arxiv_id,
            ),
            # Include annotation data if available
            terms=p.terms,
            background=p.background,
            target=p.target,
        )
        for p in papers_sampled
    ]

    # Fetch missing arXiv IDs if requested
    if fetch_arxiv:
        papers_converted, updated_count = fetch_missing_arxiv_ids(papers_converted)
    else:
        updated_count = 0

    # Print summary
    logger.info(
        "\n%s",
        display_arxiv_summary(
            len(papers_sampled), existing_arxiv_count, updated_count, fetch_arxiv
        ),
    )

    save_data(output_file, papers_converted, compress=Compress.AUTO)


if __name__ == "__main__":
    app()
