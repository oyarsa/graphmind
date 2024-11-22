"""Download papers belonging to ICLR primary areas from the Semantic Scholar API.

The primary areas are obtained from the `paper.gpt.prompt.primary_areas.toml` file.

For each primary area and year range, download the top `limit_year` papers by title
similarity to the query. The year ranges can be like `2017-2022` or single values like
`2022`. They're always strings.

---
NB: The code uses aiohttp for requests, but it's actually sequential. I tried using
concurrent requests here, but it didn't work very well.
"""

from __future__ import annotations

import asyncio
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import backoff
import dotenv
import typer
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, ConfigDict, TypeAdapter
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from paper.external_data.semantic_scholar.info import S2_SEARCH_BASE_URL
from paper.external_data.semantic_scholar.model import Paper
from paper.util import (
    arun_safe,
    die,
    display_params,
    ensure_envvar,
    progress,
    read_resource,
)

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
REQUESTS_PER_SECOND = 1
RATE_LIMITER = AsyncLimiter(1, 1)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON with the downloaded papers.")
    ],
    fields: Annotated[
        str, typer.Option(help="Command-separate list of fields to retrieve.")
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
    years: Annotated[
        Sequence[str],
        typer.Option(
            help="List of year ranges to fetch papers. Can be like `2017-2022` or `2010`."
        ),
    ] = tuple(str(y) for y in range(2012, 2021)),
    limit_year: Annotated[
        int,
        typer.Option(
            help="Number of papers per year range per primary area. Set to 0 for all."
        ),
    ] = 10,
    limit_page: Annotated[
        int,
        typer.Option(help="Number of papers per queried page.", max=100),
    ] = 100,
    limit_areas: Annotated[
        int | None,
        typer.Option(help="Number of areas to query."),
    ] = None,
    min_citations: Annotated[
        int,
        typer.Option(
            help="Minimum number of incoming citations for papers to include."
        ),
    ] = 10,
) -> None:
    arun_safe(
        download_paper_info,
        fields,
        output_file,
        years,
        limit_year,
        limit_page,
        limit_areas,
        min_citations,
    )


async def download_paper_info(
    fields_str: str,
    output_file: Path,
    year_ranges: Sequence[str],
    limit_year: int | None,
    limit_page: int,
    limit_areas: int | None,
    min_citations: int,
) -> None:
    """Download papers belonging to ICLR primary areas from the Semantic Scholar API.

    For each primary area and year range in `years`, download the top `limit_year`
    matches by similarity, as given by the API.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required, as payloads larger than 10 MB will generate
    errors. `limit_page` can be used to control the number of results per page if the
    payload size becomes a problem.
    """
    print(display_params())

    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    if not fields:
        die("No valid --fields. It should be a comma-separated strings of field names.")

    if limit_year is not None and limit_year <= 0:
        limit_year = None

    if limit_areas is not None and limit_areas <= 0:
        limit_areas = None

    if not (1 <= limit_page <= 100):
        die(f"Invalid `limit-page`: '{limit_page}'. Must be between 1 and 100.")

    primary_areas: list[str] = tomllib.loads(
        read_resource("external_data.semantic_scholar", "primary_areas.toml")
    )["primary_areas"][:limit_areas]

    area_results = await _fetch_areas(
        api_key,
        fields,
        limit_page,
        limit_year,
        primary_areas,
        year_ranges,
        min_citations,
    )

    # Print pretty table of number of retrieved papers per area
    table = Table("Area", "Number of papers")
    for area in area_results:
        table.add_row(area.area, str(len(area.papers)))
    table.add_row("Total", str(sum(len(area.papers) for area in area_results)))
    Console().print(table)

    papers = _merge_areas(area_results)
    print("Unique papers:", len(papers))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(TypeAdapter(list[PaperOutput]).dump_json(papers, indent=2))


async def _fetch_areas(
    api_key: str,
    fields: Sequence[str],
    limit_page: int,
    limit_year: int | None,
    primary_areas: Sequence[str],
    year_ranges: Sequence[str],
    min_citations: int,
) -> list[AreaResult]:
    """Fetch papers for each area and year range.

    Args:
        api_key: Semantic Scholar API key
        fields: Fields to obtain from the API.
        limit_page: Limit of papers per page request. Decrease this is the API complains
            about the payload.
        limit_year: Maximum number of papers to download for each area and year.
        primary_areas: Areas to search the API. Each area is used as the query for the
            request.
        year_ranges: Year ranges to filter the API. Can be ranges like `2017-2022` or a
            single year like `2020`.
        min_citations: Only include papers with the minimum number of citations.

    Returns:
        List of paper results per area.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        headers={"x-api-key": api_key},
        raise_for_status=True,
    ) as session:
        return [
            AreaResult(
                area=area,
                papers=await _fetch_area(
                    session,
                    area,
                    fields,
                    year_ranges,
                    limit_year=limit_year,
                    limit_page=limit_page,
                    min_citations=min_citations,
                ),
            )
            for area in tqdm(primary_areas, desc="Querying areas")
        ]


async def _fetch_area(
    session: aiohttp.ClientSession,
    query: str,
    fields: Iterable[str],
    year_ranges: Sequence[str],
    limit_year: int | None,
    limit_page: int,
    min_citations: int,
) -> list[Paper]:
    """Fetch paper information for a given `query`. Only returns data from `fields`.

    The request filters by the `query` and `year_ranges`.

    Handles pagination. Allows setting a total maximum of papers to download per year,
    and also the limit per page. The former is mostly for testing; the latter can be
    tweaked if the API is complaining the payload is too heavy. This can happen as the
    total size of the payload nears 10 MB.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        query: What we'll search in the API. This should be text of a primary area from
            the ICLR documentation.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        year_ranges: Sequence of year ranges to fetch papers. Can be like `2017-2022` or
            `2010`.
        limit_year: Maximum number of papers per year range to retrieve from the API. If
            None, retrieve as many as possible.
        limit_page: Page limit sent to the API. The API defaults to 100, but this can
            be tweaked to ensure the payloads aren't too heavy.
        min_citations: Only include papers with the minimum number of citations.

    Returns:
        List of dictionaries containing the contents of the `data` object of all pages
        of all year ranges.
    """
    tasks = [
        _fetch_area_year_range(
            session,
            query,
            fields,
            year_range,
            limit_year,
            limit_page,
            min_citations,
        )
        for year_range in year_ranges
    ]
    return [
        Paper.model_validate(paper)
        for papers in await progress.gather(tasks, desc=f"Q: {query}", leave=False)
        for paper in papers
    ]


async def _fetch_area_year_range(
    session: aiohttp.ClientSession,
    query: str,
    fields: Iterable[str],
    year_range: str,
    limit_year: int | None,
    limit_page: int,
    min_citations: int,
) -> list[dict[str, Any]]:
    """Fetch paper information for a given `query`. Only returns data from `fields`.

    Handles pagination, subject to payload sizes. See `_fetch_area` for more information.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        query: What we'll search in the API. This should be text of a primary area from
            the ICLR documentation.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        year_range: Range of years to fetch papers. Can be either like `2017-2022` or
            `2010`.
        limit_year: Maximum number of papers per year range to retrieve from the API. If
            None, retrieve as many as possible.
        limit_page: Page limit sent to the API. The API defaults to 100, but this can
            be tweaked to ensure the payloads aren't too heavy.
        min_citations: Only include papers with the minimum number of citations.

    Returns:
        List of dictionaries containing the contents of the `data` object of each page.
    """

    params = {
        "query": _clean_query(query),
        "fields": ",".join(fields),
        "limit": limit_page,
        "year": year_range,
        "minCitationCount": min_citations,
    }

    results_all: list[dict[str, Any]] = []
    offset: int | None = 0

    try:
        # Endpoint can only return up to 1000 relevance-ranked results
        while offset is not None and offset < 1000:
            async with RATE_LIMITER:
                result = await _fetch_with_retries(
                    session, params=params | {"offset": offset}, url=S2_SEARCH_BASE_URL
                )

            if error := result.get("error"):
                print(f"Error: {error}")
                return results_all

            data = result.get("data")
            if not data:
                return results_all

            results_all.extend(data)

            if limit_year is not None and len(results_all) > limit_year:
                return results_all

            offset = result.get("next")
    except Exception as e:
        print(f"Query '{query}' (last {offset=}) failed after {MAX_RETRIES} tries:")
        print("Exception:", e)

    return results_all


def _clean_query(query: str) -> str:
    """Replace anything that aren't letters or whitespace with space."""
    query = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
    return " ".join(query.split())


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
        return await response.json()


def _merge_areas(areas: Sequence[AreaResult]) -> list[PaperOutput]:
    """Merge papers across `areas` by `paper_id`, keeping track of their area sources.

    Ignores papers where the `abstract` is null or empty.
    """
    # Both have paper_id as the key
    papers: dict[str, Paper] = {}
    paper_areas: defaultdict[str, set[str]] = defaultdict(set)

    for area in areas:
        for paper in area.papers:
            if not paper.abstract:
                continue

            if paper.paper_id not in papers:
                papers[paper.paper_id] = paper

            paper_areas[paper.paper_id].add(area.area)

    return [
        PaperOutput.model_construct(
            paper_id=paper.paper_id,
            corpus_id=paper.corpus_id,
            url=paper.url,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            reference_count=paper.reference_count,
            citation_count=paper.citation_count,
            influential_citation_count=paper.influential_citation_count,
            tldr=paper.tldr,
            authors=paper.authors,
            areas=sorted(paper_areas[id]),
        )
        for id, paper in papers.items()
    ]


class AreaResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    area: str
    papers: Sequence[Paper]


class PaperOutput(Paper):
    model_config = ConfigDict(frozen=True)

    areas: Sequence[str]


if __name__ == "__main__":
    app()
