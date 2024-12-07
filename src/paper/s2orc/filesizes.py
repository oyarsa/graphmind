"""Get the size of the files from the S2ORC from the Semantic Scholar API."""

import asyncio
import json
import urllib.parse
from pathlib import Path
from typing import Annotated

import aiohttp
import dotenv
import typer

from paper.util import ensure_envvar, progress
from paper.util.cli import die

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def _get_file_size(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> int:
    """Get the file size from the given URL.

    Respects `semaphore` to limit the number of concurrent requests.
    """
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


def _bytes_to_gib(bytes_size: int) -> float:
    return bytes_size / (1024 * 1024 * 1024)


async def _get_filesizes(
    dataset_name: str, output_file: Path, api_key: str, limit: int | None
) -> None:
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
        Path("dataset.json").write_text(json.dumps(dataset, indent=2))

        if "files" not in dataset or not dataset["files"]:
            die("No files found.")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        files = dataset["files"][:limit]
        tasks = [_get_file_size(url, session, semaphore) for url in files]
        file_sizes = await progress.gather(tasks, desc="Getting file sizes")

        total_size_gb = sum(_bytes_to_gib(size) for size in file_sizes)
        info: list[dict[str, str | float]] = []

        print("\nFile sizes:")
        for url, size in zip(files, file_sizes):
            file_name = urllib.parse.urlparse(str(url)).path.split("/")[-1]
            size_gb = _bytes_to_gib(size)
            print(f"{file_name}: {size_gb:.2f} GiB")

            info.append({"url": url, "name": file_name, "size_gb": size_gb})

        print(f"\nTotal size of all files: {total_size_gb:.2f} GiB")

        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(json.dumps(info, indent=2))


def get_filesizes(dataset_name: str, output_path: Path, limit: int | None) -> None:
    """Get the size of the files from the dataset from the Semantic Scholar API."""

    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    asyncio.run(_get_filesizes(dataset_name, output_path, api_key, limit))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    dataset_name: Annotated[
        str, typer.Argument(help="Name of the dataset for which to get the file sizes.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to JSON file with file information.")
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            typer.Option(
                help="Limit the number of files to download. Useful for testing."
            ),
        ),
    ] = None,
) -> None:
    """Get the size of the files from the S2ORC from the Semantic Scholar API."""
    get_filesizes(dataset_name, output_file, limit)


if __name__ == "__main__":
    app()
