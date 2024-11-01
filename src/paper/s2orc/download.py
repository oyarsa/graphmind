"""Download files from the S2ORC from the Semantic Scholar API."""

import argparse
import asyncio
import json
import os
import sys
import urllib.parse
from collections.abc import Coroutine
from pathlib import Path

import aiohttp
import dotenv
from tqdm.asyncio import tqdm

from paper.progress import gather

MAX_CONCURRENT_DOWNLOADS = 10
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout for each file
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def _try_download_file(
    url: str, session: aiohttp.ClientSession, display_path: Path, part_path: Path
) -> None:
    """Actually download the file in chunks and handle progress bar.

    Handles timeouts, but not retries. This function should be called from a retry loop.
    This will throw if something goes wrong (HTTP error, timeout, etc.)
    """
    async with session.get(
        url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    ) as response:
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(part_path, "wb") as file,
            tqdm(
                desc=str(display_path),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            async for chunk in response.content.iter_chunked(1024):
                size = file.write(chunk)
                progress_bar.update(size)


async def _download_file(
    url: str, path: Path, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> None:
    """Download a file from the given URL with a human-readable progress bar.
    The file is first downloaded to a .part file and then renamed.

    This function is a wrapper around _download_file that handles retries and errors.
    """
    async with semaphore:
        part_path = path.with_suffix(path.suffix + ".part")

        for attempt in range(MAX_RETRIES):
            try:
                await _try_download_file(url, session, path, part_path)
            except Exception as e:
                print(
                    f"Error downloading {path}: {e}. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
            else:
                # If download completes successfully, rename the file
                part_path.rename(path)
                return

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)

        print(f"Failed to download {path} after {MAX_RETRIES} attempts.")


async def _download(
    dataset_name: str, output_path: Path, api_key: str, limit: int | None
) -> None:
    """Actually download dataset files from the Semantic Scholar API to the output path.

    Handles the async calls, sessions and so on.
    """
    async with aiohttp.ClientSession() as session:
        # Get latest release's ID
        async with session.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        ) as response:
            release_id = (await response.json())["release_id"]
        print(f"Latest release ID: {release_id}")

        # Get the download links for the s2orc dataset
        async with session.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}/",
            headers={"x-api-key": api_key},
        ) as response:
            dataset = await response.json()
        (output_path / "dataset.json").write_text(json.dumps(dataset, indent=2))

        if "files" not in dataset or not dataset["files"]:
            print("No files found.")
            sys.exit(1)

        output_path.mkdir(exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        tasks: list[Coroutine[None, None, None]] = []

        for url in dataset["files"][:limit]:
            file_name = urllib.parse.urlparse(str(url)).path.split("/")[-1]
            file_path = output_path / file_name

            tasks.append(_download_file(url, file_path, session, semaphore))

        await gather(tasks, desc="Overall progress")


def download_dataset(
    dataset_name: str, output_path: Path, api_key: str | None, limit: int | None
) -> None:
    """Download dataset files from the Semantic Scholar API to the output path.

    Prevents the user from accidentally exiting the script with Ctrl+C. Instead, it asks
    for confirmation before exiting.
    """
    dotenv.load_dotenv()
    if api_key is None:
        api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

    output_path.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            asyncio.run(_download(dataset_name, output_path, api_key, limit))
            break  # If _download completes without interruption, exit the loop
        except KeyboardInterrupt:
            choice = input("\n\nCtrl+C detected. Do you really want to exit? (y/n): ")
            if choice.lower() == "y":
                sys.exit()
            else:
                # The loop will continue, restarting _download
                print("Continuing...\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the downloaded files"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="s2orc",
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the Semantic Scholar API. Defaults to the SEMANTIC_SCHOLAR_API_KEY environment variable.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit the number of files to download. Useful for testing.",
    )
    args = parser.parse_args()
    download_dataset(args.dataset_name, args.output_path, args.api_key, args.limit)


if __name__ == "__main__":
    main()
