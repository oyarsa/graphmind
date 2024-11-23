"""Download information for papers referenced in ASAP using the Semantic Scholar API.

This has two modes:
- main: Take the titles of the ASAP papers and query the API.
- references: Gather the unique papers referenced by the ASAP papers and query their
  titles.

The output we get from the API is the same in both cases.

The input is the output of the ASAP pipeline[1], asap_filtered.json.

S2 does their own paper title matching, so we just take the best match directly. We then
save the entire output to a JSON file along with the original query title, which might
be different.

The resulting files are:
- semantic_scholar_full.json: the whole output from the S2 API.
- semantic_scholar_valid.json: the valid (non-empty) results with fuzzy ratio.
- semantic_scholar_final.json: the valid results with a non-empty abstract
  and fuzzy ratio above a minium (default: 80).

[1] See paper.asap.preprocess.
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import dotenv
import typer

from paper.external_data.semantic_scholar.model import title_ratio
from paper.util import arun_safe, ensure_envvar, progress, setup_logging

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
RETRY_DELAY = 5  # Initial delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff factor

S2_SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

logger = logging.getLogger(__name__)


async def _fetch_paper_info(
    session: aiohttp.ClientSession,
    api_key: str,
    paper_title: str,
    fields: Sequence[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch paper information for a given title. Takes only the best title match."""
    params = {
        "query": paper_title,
        "fields": ",".join(fields),
        "limit": 1,  # We're only interested in the best match
    }
    headers = {"x-api-key": api_key}

    async with semaphore:
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
                            return {"title_query": paper_title} | data["data"][0]
                        logger.debug(f"No results found for title: {paper_title}")
                        return None

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
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


class InfoMode(StrEnum):
    MAIN = "main"
    REFERENCES = "references"


async def _download_paper_info(
    input_file: Path,
    mode: InfoMode,
    fields_str: str,
    output_path: Path,
    api_key: str,
    min_fuzzy: int | None,
) -> None:
    """Download paper information for multiple titles.

    The input file is the output of the ASAP preprocessing pipeline (asap_filtered.json).
    We obtain the unique titles from the references of all papers and get their info.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required.
    """
    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    papers: list[dict[str, Any]] = json.loads(input_file.read_text())

    match mode:
        case InfoMode.MAIN:
            unique_titles: set[str] = {paper["title"] for paper in papers}
        case InfoMode.REFERENCES:
            unique_titles: set[str] = {
                reference["title"]
                for paper in papers
                for reference in paper["references"]
            }

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
    ) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            _fetch_paper_info(session, api_key, title, fields, semaphore)
            for title in unique_titles
        ]
        results = list(await progress.gather(tasks, desc="Downloading paper info"))

    results_valid = [
        result | {"fuzz_ratio": title_ratio(result["title_query"], result["title"])}
        for result in results
        if result
    ]
    results_filtered = [
        paper
        for paper in results_valid
        if paper["fuzz_ratio"] >= min_fuzzy and paper["abstract"]
    ]

    logger.info(len(results), "papers")
    logger.info(len(results_valid), "valid")
    logger.info(
        len(results_filtered),
        f"filtered (non-emtpy abstract and fuzz ratio >= {min_fuzzy})",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "semantic_scholar_full.json").write_text(
        json.dumps(results, indent=2)
    )
    (output_path / "semantic_scholar_valid.json").write_text(
        json.dumps(results_valid, indent=2)
    )
    (output_path / "semantic_scholar_final.json").write_text(
        json.dumps(results_filtered, indent=2)
    )


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
        Path, typer.Argument(help="Input file (e.g. asap_filtered.json).")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded information.")
    ],
    mode: Annotated[
        InfoMode,
        typer.Option(help="Download information from main papers or their references."),
    ] = InfoMode.REFERENCES,
    fields: Annotated[
        str,
        typer.Option(help="Comma-separated list of fields to retrieve."),
    ] = ",".join(
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
            "tldr",
        )
    ),
    min_fuzzy: Annotated[
        int, typer.Option(help="Minimum fuzz ratio of titles to filter.", max=100)
    ] = 80,
) -> None:
    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    setup_logging()

    arun_safe(
        _download_paper_info,
        input_file,
        mode,
        fields,
        output_path,
        api_key,
        min_fuzzy,
    )


if __name__ == "__main__":
    app()
