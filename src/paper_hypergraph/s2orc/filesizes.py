"""Get the size of the files from the S2ORC from the Semantic Scholar API."""

import asyncio
import json
import os
import sys
import urllib.parse
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

MAX_FILES = None
API_KEY = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
DATASET_NAME = "s2orc"
LOCAL_PATH = Path("./data")

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def get_file_size(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> int:
    """Get the file size from the given URL."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as response:
                    size = int(response.headers.get("Content-Length", 0))
                    if size == 0:
                        # If Content-Length is not provided, read the entire content
                        content = await response.read()
                        size = len(content)
                    return size
            except Exception as e:
                print(
                    f"Error getting file size for {url}: {e}. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)

        print(f"Failed to get file size for {url} after {MAX_RETRIES} attempts.")
        return 0


def bytes_to_gib(bytes_size: int) -> float:
    return bytes_size / (1024 * 1024 * 1024)


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        # Get latest release's ID
        async with session.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        ) as response:
            release_id = (await response.json())["release_id"]
        print(f"Latest release ID: {release_id}")

        # Get the download links for the s2orc dataset
        async with session.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{DATASET_NAME}/",
            headers={"x-api-key": API_KEY},
        ) as response:
            dataset = await response.json()
        Path("dataset.json").write_text(json.dumps(dataset, indent=2))

        if "files" not in dataset or not dataset["files"]:
            print("No files found.")
            sys.exit(1)

        LOCAL_PATH.mkdir(exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        tasks = [
            get_file_size(url, session, semaphore)
            for url in dataset["files"][:MAX_FILES]
        ]
        file_sizes = await tqdm.gather(*tasks, desc="Getting file sizes")

        total_size_gb = sum(bytes_to_gib(size) for size in file_sizes)

        print("\nFile sizes:")
        for url, size in zip(dataset["files"][:MAX_FILES], file_sizes):
            file_name = urllib.parse.urlparse(url).path.split("/")[-1]
            print(f"{file_name}: {bytes_to_gib(size):.2f} GiB")

        print(f"\nTotal size of all files: {total_size_gb:.2f} GiB")


if __name__ == "__main__":
    asyncio.run(main())
