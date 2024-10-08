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

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
RETRY_DELAY = 5  # Initial delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff factor


async def _fetch_paper_info(
    session: aiohttp.ClientSession,
    api_key: str,
    title: str,
    fields: Sequence[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Fetch paper information for a given title."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
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
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data"):
                            return data["data"][0]
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
    titles: Sequence[str], fields: Sequence[str], output_path: Path, api_key: str
) -> None:
    """Download paper information for multiple titles."""
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=connector,
    ) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            _fetch_paper_info(session, api_key, title, fields, semaphore)
            for title in titles
        ]
        results = await progress_gather(*tasks, desc="Downloading paper info")

    valid_results = [result for result in results if result]
    output_file = output_path / "paper_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)

    print(
        f"Downloaded information for {len(valid_results)} papers. Saved to {output_file}"
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
    args = parser.parse_args()

    dotenv.load_dotenv()
    api_key = args.api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print(
            "Error: No API key provided. Please set the SEMANTIC_SCHOLAR_API_KEY"
            " environment variable or use the --api-key argument."
        )
        sys.exit(1)

    args.output_path.mkdir(parents=True, exist_ok=True)

    titles = [
        title
        for line in args.input_file.read_text(encoding="utf-8").splitlines()
        if (title := line.strip())
    ]
    fields = args.fields.split(",")

    while True:
        try:
            asyncio.run(_download_paper_info(titles, fields, args.output_path, api_key))
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
