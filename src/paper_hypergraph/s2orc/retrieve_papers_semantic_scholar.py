"""Download paper information from the Semantic Scholar API."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiohttp
import dotenv

from paper_hypergraph.s2orc.download import progress_gather
from paper_hypergraph.util import fuzzy_ratio

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
RETRY_DELAY = 5  # Initial delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff factor


SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


async def _fetch_paper_info(
    session: aiohttp.ClientSession,
    api_key: str,
    paper: dict[str, str],
    fields: Sequence[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch paper information for a given title."""
    title = paper["title"]
    params = {
        "query": title,
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
                    SEMANTIC_SCHOLAR_BASE_URL, params=params, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data"):
                            return {"title_query": title} | data["data"][0]
                        else:
                            print(f"No results found for title: {title}")
                            return None
                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                            wait_source = "Retry-After"
                        else:
                            wait_time = delay
                            wait_source = "Custom"
                        print(
                            f"Rate limited (429) when fetching '{title}'. "
                            f"Retrying after {wait_time} ({wait_source}) seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        delay *= BACKOFF_FACTOR  # Exponential backoff
                    else:
                        error_text = await response.text()
                        print(
                            f"Error fetching data for '{title}': HTTP {response.status} - {error_text}"
                        )
                        return None
            except aiohttp.ClientError as e:
                print(
                    f"Network error fetching '{title}': {e}. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR
            except TimeoutError:
                print(
                    f"Timeout error fetching '{title}'. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR

            attempt += 1

        print(f"Failed to fetch data for '{title}' after {MAX_RETRIES} attempts.")
        return None


async def _download_paper_info(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    api_key: str,
    min_fuzzy: int | None,
) -> None:
    """Download paper information for multiple titles."""
    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    papers = json.loads(input_file.read_text())

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
    ) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            _fetch_paper_info(session, api_key, paper, fields, semaphore)
            for paper in papers
        ]
        results = list(await progress_gather(*tasks, desc="Downloading paper info"))

    results = [
        result | {"fuzz_ratio": fuzzy_ratio(result["title_query"], result["title"])}
        for result in results
    ]
    results_valid = [result for result in results if result]

    print(len(results), "papers")
    print(len(results_valid), "valid")

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "semantic_scholar_full.json").write_text(
        json.dumps(results, indent=2)
    )
    (output_path / "semantic_scholar_best.json").write_text(
        json.dumps(results_valid, indent=2)
    )

    if min_fuzzy:
        filtered_data = [
            paper
            for paper in results_valid
            if paper["fuzz_ratio"] >= min_fuzzy and paper["abstract"]
        ]
        (output_path / "semantic_scholar_filtered.json").write_text(
            json.dumps(filtered_data, indent=2)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=Path, help="File containing paper titles, one per line"
    )
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the downloaded information"
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="title,authors,year,abstract",
        help="Comma-separated list of fields to retrieve",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the Semantic Scholar API. Defaults to the"
        " SEMANTIC_SCHOLAR_API_KEY environment variable.",
    )
    parser.add_argument(
        "--min-fuzzy",
        type=int,
        default=None,
        help="Minimum fuzz ratio of titles to filter",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    api_key = args.api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print(
            "Error: No API key provided. Please set the SEMANTIC_SCHOLAR_API_KEY"
            " environment variable or use the --api-key argument."
        )
        sys.exit(1)

    while True:
        try:
            asyncio.run(
                _download_paper_info(
                    args.input_file,
                    args.fields,
                    args.output_path,
                    api_key,
                    args.min_fuzzy,
                )
            )
            break  # If _download completes without interruption, exit the loop
        except KeyboardInterrupt:
            choice = input("\n\nCtrl+C detected. Do you really want to exit? (y/n): ")
            if choice.lower() == "y":
                sys.exit()
            else:
                # The loop will continue, restarting _download
                print("Continuing...\n")


if __name__ == "__main__":
    main()
