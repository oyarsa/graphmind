"""Download papers belonging to ICLR primary areas from the Semantic Scholar API.

The primary areas are obtained from the `paper.gpt.prompt.primary_areas.toml` file.

For each primary area, download the top `limit_papers` papers by title similarity to the
query.
"""

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
        "--limit-total",
        type=int,
        default=10,
        help="Number of papers per primary area to download. Set to 0 for all.",
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
        args.limit_total,
        args.limit_page,
    )


async def download_paper_info(
    fields_str: str, output_path: Path, limit_total: int | None, limit_page: int
) -> None:
    """Download papers belonging to ICLR primary areas from the S2 API.

    For each primary area, download the top `limit_total` papers by "influential
    citations".

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required, as payloads larger than 10 MB will generate
    errors.
    """
    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    if not fields:
        sys.exit(
            "No valid --fields. It should be a comma-separated strings of field names."
        )

    if limit_total is not None and limit_total <= 0:
        limit_total = None

    if not (1 <= limit_page <= 100):
        sys.exit(f"Invalid `limit-page`: '{limit_page}'. Must be between 1 and 100.")

    primary_areas: list[str] = tomllib.loads(
        read_resource("gpt.prompts", "primary_areas.toml")
    )["primary_areas"]

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
        headers={"x-api-key": api_key},
        raise_for_status=True,
    ) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        results = [
            AreaResult(
                area=area,
                papers=await _fetch_area(
                    session,
                    area,
                    fields,
                    semaphore,
                    limit_total=limit_total,
                    limit_page=limit_page,
                ),
            )
            for area in primary_areas
        ]

    table = Table("Area", "Number of papers")
    for area in results:
        table.add_row(area.area, str(len(area.papers)))
    table.add_row("Total", str(sum(len(area.papers) for area in results)))
    Console().print(table)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "primary_area_papers.json").write_bytes(
        TypeAdapter(list[AreaResult]).dump_json(results, indent=2)
    )


async def _fetch_area(
    session: aiohttp.ClientSession,
    query: str,
    fields: Iterable[str],
    semaphore: asyncio.Semaphore,
    limit_total: int | None,
    limit_page: int,
    url: str = S2_SEARCH_BASE_URL,
) -> list[dict[str, Any]]:
    """Fetch paper information for a given `query`. Only returns data from `fields`.

    Handles pagination. Allows setting a total maximum of papers to download, and also
    the limit per page. The former is mostly for testing; the latter can be tweaked
    if the API is complaining the payload is too heavy. This can have happen as the
    total size of the payload nears 10 MB.

    Args:
        session: Client session. Assumes a the API key has been set in the headers.
        query: What we'll search in the API. This should be text of a primary area from
            the ICLR documentation.
        fields: List of fields to retrieve. Restrict this only to the bare essentials
            to ensure the payloads are lightweight.
        semaphore: Lock used to ensure that not too many requests are made at the same
            time.
        limit_total: Maximum number of papers to retrieve from the API. If None, retrieve
            as many as possible.
        limit_page: Page limit sent to the API. The API defaults to 100, but this can
            be tweaked to ensure the payloads aren't too heavy.
        url: base URL for the 'Paper relevance search' endpoint.

    Returns:
        List of dictionaries containing the contents of the `data` object of each page.
        No modification is done to these items.
    """

    params = {"query": query, "fields": ",".join(fields), "limit": limit_page}

    results_all: list[dict[str, Any]] = []
    offset: int | None = 0

    try:
        while offset is not None:
            async with semaphore:
                result = await _fetch_with_retries(
                    session, params=params | {"offset": offset}, url=url
                )

            data = result.get("data")
            if not data:
                print(f"Query '{query}': result.data is unavailable or empty.")
                return results_all

            results_all.extend(data)

            if limit_total is not None and len(results_all) > limit_total:
                print(f"Query '{query}': paper limit ({limit_total}) reached.")
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
