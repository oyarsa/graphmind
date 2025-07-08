"""Download information for PeerRead papers and references from the Semantic Scholar API.

This has two modes:
- main: Take the titles of the PeerRead papers and query the API.
- references: Gather the unique papers referenced by the PeerRead papers and query their
  titles.

The output we get from the API is the same in both cases, but the resulting files have
different shapes.

The input is the output of the PeerRead pipeline[1], peerread_merged.json.

S2 does their own paper title matching, so we just take the best match directly. We then
save the entire output to a JSON file along with the original query title, which might
be different.

Note that the API allows us to specify what fields we want. It's best to minimise the
number of fields so we don't run into bandwidth issues.

The resulting files in both modes are:
- valid.json: the valid (non-empty) results with fuzzy ratio.
- final.json: the valid results with a non-empty abstract and fuzzy ratio above a
  minimum (default: 80).

The output types are:
- main: s2.PeerPaperWithS2 (combines the input PeerRead paper and its S2 information).
- references: peerread.S2Paper (only the reference S2 information with the query title).

[1] See paper.peerread.process.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import dotenv
import typer
from pydantic import ValidationError
from tqdm import tqdm

from paper import embedding as emb
from paper import peerread as pr
from paper.semantic_scholar.model import (
    PaperFromPeerRead,
    PeerReadPaperWithS2,
    title_ratio,
)
from paper.semantic_scholar.recommended import Limiter
from paper.util import arun_safe, ensure_envvar, progress, setup_logging
from paper.util.serde import load_data, save_data

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
RETRY_DELAY = 5  # Initial delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff factor

S2_SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

logger = logging.getLogger("paper.semantic_scholar.info")


async def fetch_paper_data(
    session: aiohttp.ClientSession,
    api_key: str,
    paper_title: str,
    fields: Sequence[str],
    limiter: Limiter,
) -> dict[str, Any] | None:
    """Fetch raw paper data from the API with retry logic."""
    params = {
        "query": paper_title,
        "fields": ",".join(fields),
        "limit": 1,  # We're only interested in the best match
    }
    headers = {"x-api-key": api_key}

    async with limiter:
        attempt = 0
        delay = RETRY_DELAY
        while attempt < MAX_RETRIES:
            try:
                async with session.get(
                    S2_SEARCH_BASE_URL, params=params, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data"):
                            return data["data"][0] | {"title_query": paper_title}
                        logger.debug(f"No results found for title: {paper_title}")
                        return None

                    if response.status == 429:
                        if retry_after := response.headers.get("Retry-After"):
                            wait_time = int(retry_after)
                            wait_source = "Retry-After"
                        else:
                            wait_time = delay
                            wait_source = "Custom"
                        logger.debug(
                            f"Rate limited (429) when fetching '{paper_title}'. "
                            f"Retrying after {wait_time} ({wait_source}) seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        delay *= BACKOFF_FACTOR  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Error fetching data for '{paper_title}': HTTP {response.status} - {error_text}"
                        )
                        return None
            except aiohttp.ClientError as e:
                logger.debug(
                    f"Network error fetching '{paper_title}': {e}. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR
            except TimeoutError:
                logger.debug(
                    f"Timeout error fetching '{paper_title}'. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR

            attempt += 1

        logger.warning(
            f"Failed to fetch data for '{paper_title}' after {MAX_RETRIES} attempts."
        )
        return None


async def fetch_paper_info(
    session: aiohttp.ClientSession,
    api_key: str,
    paper_title: str,
    fields: Sequence[str],
    limiter: Limiter,
) -> PaperFromPeerRead | None:
    """Fetch paper information for a given title. Takes only the best title match.

    The title match is done by the S2 API, not us. It's usually pretty good.
    """
    raw_data = await fetch_paper_data(session, api_key, paper_title, fields, limiter)
    if raw_data is None:
        return None

    try:
        return PaperFromPeerRead.model_validate(raw_data)
    except ValidationError as e:
        logger.debug("Validation error: %s.", e)
        return None


async def fetch_papers_from_s2(
    api_key: str,
    titles: Sequence[str],
    fields: Sequence[str],
    *,
    desc: str | None = None,
    limiter: Limiter | None = None,
) -> list[PaperFromPeerRead | None]:
    """Fetch paper information for multiple titles in parallel.

    Args:
        api_key: Semantic Scholar API key.
        titles: Paper titles to search for.
        fields: Fields to retrieve from S2 API.
        desc: If provided, used as the description for a progress bar.
        limiter: If provided, Semaphore/rate limiter for the requests. If not, creates
            a semaphore with MAX_CONCURRENT_REQUESTS.

    Returns:
        List of papers found (includes None for failed/not found papers).
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
    ) as session:
        limiter = limiter or asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            fetch_paper_info(session, api_key, title, fields, limiter)
            for title in titles
        ]
        if desc:
            return list(await progress.gather(tasks, desc=desc))
        else:
            return await asyncio.gather(*tasks)


async def _download_main_info(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    min_fuzzy: int,
    limit_papers: int | None,
) -> None:
    """Download paper information for main PeerRead papers.

    We obtain the unique titles from the references of all papers and get their info. We
    store the original PeerRead information alongside the retrieved S2 information.
    """
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    if limit_papers == 0:
        limit_papers = None

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    papers = load_data(input_file, pr.Paper)[:limit_papers]
    title_to_paper = {paper.title: paper for paper in papers}

    results = await fetch_papers_from_s2(
        api_key, list(title_to_paper.keys()), fields, desc="Downloading paper info"
    )

    results_valid = [
        PeerReadPaperWithS2.from_peer(title_to_paper[title_peer], s2_paper)
        for s2_paper, title_peer in zip(results, title_to_paper)
        if s2_paper
    ]

    results_filtered = [
        paper for paper in results_valid if paper.fuzz_ratio >= min_fuzzy
    ]

    logger.info(f"{len(results)} papers")
    logger.info(f"{len(results_valid)} valid")
    logger.info(
        f"{len(results_filtered)} filtered (non-emtpy abstract and fuzz ratio >="
        f" {min_fuzzy})",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    save_data(output_path / "valid.json.zst", results_valid)
    save_data(output_path / "final.json.zst", results_filtered)


async def _download_reference_info(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    min_fuzzy: int,
    limit_papers: int | None,
    top_k: int | None,
) -> None:
    """Download paper information for reference papers."""
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    if limit_papers == 0:
        limit_papers = None

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    papers = load_data(input_file, pr.Paper)[:limit_papers]

    if top_k == 0:
        top_k = None

    if top_k is None:
        titles = [ref.title for paper in papers for ref in paper.references]
    else:
        encoder = emb.Encoder()
        titles = [
            get_top_k_titles(encoder, paper, top_k)
            for paper in tqdm(papers, desc=f"Filtering top {top_k} references")
        ]

    unique_titles = {title for titles in titles for title in titles}
    logger.info(f"{len(unique_titles)} unique titles")

    results = await fetch_papers_from_s2(
        api_key, list(unique_titles), fields, desc="Downloading paper info"
    )

    results_valid = [paper for paper in results if paper]

    results_filtered = [
        paper
        for paper in results_valid
        if title_ratio(paper.title, paper.title_peer) >= min_fuzzy
    ]

    logger.info(f"{len(results)} papers")
    logger.info(f"{len(results_valid)} valid")
    logger.info(
        f"{len(results_filtered)} filtered (non-emtpy abstract and fuzz ratio >="
        f" {min_fuzzy})",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    save_data(output_path / "valid.json.zst", [paper for paper in results if paper])
    save_data(output_path / "final.json.zst", results_filtered)


def get_top_k_titles(encoder: emb.Encoder, paper: pr.Paper, k: int) -> list[str]:
    """Get top `k` reference titles from `paper`.

    References are sorted by cosine similarity between the reference and main paper
    titles.
    """
    if not paper.references:
        return []

    ref_titles = [r.title for r in paper.references]

    references_emb = encoder.batch_encode(ref_titles)
    title_emb = encoder.encode(paper.title)
    sim = emb.similarities(title_emb, references_emb)

    return [ref_titles[idx] for idx in emb.top_k_indices(sim, k)]


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def main(
    input_file: Annotated[
        Path, typer.Argument(help="Input file (e.g. peerread_merged.json).")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded information.")
    ],
    fields: Annotated[
        str,
        typer.Option(help="Comma-separated list of fields to retrieve."),
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
        "tldr",
    )),
    min_fuzzy: Annotated[
        int, typer.Option(help="Minimum fuzz ratio of titles to filter.", max=100)
    ] = 80,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit on the number of papers to query. Use 0 for all.",
        ),
    ] = None,
) -> None:
    """Download paper information for PeerRead main papers."""

    arun_safe(_download_main_info, input_file, fields, output_path, min_fuzzy, limit)


@app.command(no_args_is_help=True)
def references(
    input_file: Annotated[
        Path, typer.Argument(help="Input file (e.g. peerread_merged.json).")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded information.")
    ],
    fields: Annotated[
        str,
        typer.Option(help="Comma-separated list of fields to retrieve."),
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
        "tldr",
    )),
    min_fuzzy: Annotated[
        int, typer.Option(help="Minimum fuzz ratio of titles to filter.", max=100)
    ] = 80,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit on the number of papers to query. Use 0 for all.",
        ),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="How many references to query per paper, sorted by semantic similarity."
            " Use 0 to query all.",
        ),
    ] = 20,
) -> None:
    """Download paper information for reference papers.

    For each paper, we take only the top K references by title semantic similarity if
    `top_k` is not 0 or None.

    We obtain the unique titles from the references of all papers and get their info. We
    also store the query title.
    """
    arun_safe(
        _download_reference_info,
        input_file,
        fields,
        output_path,
        min_fuzzy,
        limit,
        top_k,
    )


@app.callback(help=__doc__)
def doc() -> None:
    """Documentation callback."""
    dotenv.load_dotenv()
    setup_logging()


if __name__ == "__main__":
    app()
