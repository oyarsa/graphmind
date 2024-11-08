"""Download papers belonging to ICLR primary areas from the Semantic Scholar API.

The primary areas are obtained from the `paper.gpt.prompt.primary_areas.toml` file.

For each primary area and year, download the top `limit_year` papers by title similarity
to the query. The years can be ranges like `2017-2022` or single values like `2022`.
They're always strings.
"""

from __future__ import annotations

import asyncio
import sys
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import aiohttp
import backoff
import dotenv
from pydantic import BaseModel, ConfigDict, TypeAdapter
from rich.console import Console
from rich.table import Table

from paper import progress
from paper.external_data.semantic_scholar_info import S2_SEARCH_BASE_URL
from paper.util import (
    HelpOnErrorArgumentParser,
    arun_safe,
    ensure_envvar,
    read_resource,
)

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the downloaded papers."
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
                "tldr",
            )
        ),
        help="Comma-separated list of fields to retrieve.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=str,
        default=["2017"],
        help="List of year ranges to fetch papers. Can be like `2017-2022` or `2010`",
    )
    parser.add_argument(
        "--limit-year",
        type=int,
        default=10,
        help="Number of papers per year per primary area. Set to 0 for all.",
    )
    parser.add_argument(
        "--limit-page",
        type=int,
        default=100,
        help="Number of papers per queried page. Must be <=100.",
    )
    args = parser.parse_args()

    arun_safe(
        download_paper_info,
        args.fields,
        args.output_path,
        args.years,
        args.limit_year,
        args.limit_page,
    )


async def download_paper_info(
    fields_str: str,
    output_path: Path,
    years: Sequence[str],
    limit_year: int | None,
    limit_page: int,
) -> None:
    """Download papers belonging to ICLR primary areas from the Semantic Scholar API.

    For each primary area and year ranges in `years`, download the top `limit_year`
    matches by similarity, as given by the API.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required, as payloads larger than 10 MB will generate
    errors. `limit_page` can be used to control the number of results per page if the
    payload size becomes a problem.
    """
    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    if not fields:
        sys.exit(
            "No valid --fields. It should be a comma-separated strings of field names."
        )

    if limit_year is not None and limit_year <= 0:
        limit_year = None

    if not (1 <= limit_page <= 100):
        sys.exit(f"Invalid `limit-page`: '{limit_page}'. Must be between 1 and 100.")

    primary_areas: list[str] = tomllib.loads(
        read_resource("gpt.prompts", "primary_areas.toml")
    )["primary_areas"]

    area_results = await _fetch_areas(
        api_key, fields, limit_page, limit_year, primary_areas, years
    )

    # Print pretty table of number of retrieved papers per area
    table = Table("Area", "Number of papers")
    for area in area_results:
        table.add_row(area.area, str(len(area.papers)))
    table.add_row("Total", str(sum(len(area.papers) for area in area_results)))
    Console().print(table)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "primary_area_papers.json").write_bytes(
        TypeAdapter(list[AreaResult]).dump_json(area_results, indent=2)
    )


async def _fetch_areas(
    api_key: str,
    fields: Sequence[str],
    limit_page: int,
    limit_year: int | None,
    primary_areas: Sequence[str],
    years: Sequence[str],
) -> list[AreaResult]:
    """Fetch papers for each area, for each year ranges.

    Args:
        api_key: Semantic Scholar API key
        fields: Fields to obtain from the API.
        limit_page: Limit of papers per page request. Decrease this is the API complains
            about the payload.
        limit_year: Maximum number of papers to download for each area and year.
        primary_areas: Areas to search the API. Each area is used as the query for the
            request.
        years: Years to filter the API. Can be ranges like `2017-2022` or a single year
            like `2020`.

    Returns:
        List of paper results per area.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
        headers={"x-api-key": api_key},
        raise_for_status=True,
    ) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            _fetch_area(
                session,
                area,
                fields,
                years,
                semaphore,
                limit_year=limit_year,
                limit_page=limit_page,
            )
            for area in primary_areas
        ]
        task_results = await progress.gather(tasks, desc="Retrieving areas")
        area_results = [
            AreaResult(area=area, papers=papers)
            for area, papers in zip(primary_areas, task_results)
        ]
    return area_results


async def _fetch_area(
    session: aiohttp.ClientSession,
    query: str,
    fields: Iterable[str],
    years: Sequence[str],
    semaphore: asyncio.Semaphore,
    limit_year: int | None,
    limit_page: int,
) -> list[dict[str, Any]]:
    """Fetch paper information for a given `query`. Only returns data from `fields`.

    The request filters by the `query` and `years`.

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
        years: Sequence of year ranges to fetch papers. Can be like `2017-2022` or
            `2010`.
        semaphore: Lock used to ensure that not too many requests are made at the same
            time.
        limit_year: Maximum number of papers per year to retrieve from the API. If None,
            retrieve as many as possible.
        limit_page: Page limit sent to the API. The API defaults to 100, but this can
            be tweaked to ensure the payloads aren't too heavy.

    Returns:
        List of dictionaries containing the contents of the `data` object of all pages
        of all years.
    """
    tasks = [
        _fetch_area_year(
            session, query, fields, year, semaphore, limit_year, limit_page
        )
        for year in years
    ]
    results = await progress.gather(tasks, desc=f"Retrieving query: '{query}'")
    return [paper for papers in results for paper in papers]


async def _fetch_area_year(
    session: aiohttp.ClientSession,
    query: str,
    fields: Iterable[str],
    year: str,
    semaphore: asyncio.Semaphore,
    limit_year: int | None,
    limit_page: int,
) -> list[dict[str, Any]]:
    """Fetch paper information for a given `query`. Only returns data from `fields`.

    Handles pagination, subject to payload sizes. See `_fetch_area` for more information.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        query: What we'll search in the API. This should be text of a primary area from
            the ICLR documentation.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        year: Range of years to fetch papers. Can be either like `2017-2022` or `2010`.
        semaphore: Lock used to ensure that not too many requests are made at the same
            time.
        limit_year: Maximum number of papers per year to retrieve from the API. If None,
            retrieve as many as possible.
        limit_page: Page limit sent to the API. The API defaults to 100, but this can
            be tweaked to ensure the payloads aren't too heavy.
        url: base URL for the 'Paper relevance search' endpoint.

    Returns:
        List of dictionaries containing the contents of the `data` object of each page.
    """

    params = {
        "query": query,
        "fields": ",".join(fields),
        "limit": limit_page,
        "year": year,
    }

    results_all: list[dict[str, Any]] = []
    offset: int | None = 0

    try:
        while offset is not None:
            async with semaphore:
                result = await _fetch_with_retries(
                    session, params=params | {"offset": offset}, url=S2_SEARCH_BASE_URL
                )

            data = result.get("data")
            if not data:
                print(f"Query '{query}': result.data is unavailable or empty.")
                return results_all

            results_all.extend(data)

            if limit_year is not None and len(results_all) > limit_year:
                print(f"Query '{query}' - {year}: paper limit ({limit_year}) reached.")
                return results_all

            offset = result.get("offset")
    except Exception as e:
        print(f"Query '{query}' (last {offset=}) failed after {MAX_RETRIES} tries:")
        print(e)

    return results_all


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


class AreaResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    area: str
    papers: Sequence[dict[str, Any]]


if __name__ == "__main__":
    main()
