"""List and download Semantic Scholar datasets."""

import asyncio
import io
import json
import urllib.parse
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Self

import aiohttp
import typer
from tqdm.asyncio import tqdm

from paper.util import arun_safe, dotenv, ensure_envvar, progress
from paper.util.cli import die
from paper.util.serde import write_file_bytes

MAX_CONCURRENT_DOWNLOADS = 10
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout for each file
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(name="list")
def list_datasets(
    show_json: Annotated[
        bool,
        typer.Option(
            "--json", help="Output data in JSON format instead of plain text."
        ),
    ] = False,
) -> None:
    """List available datasets."""
    asyncio.run(_list_datasets(show_json))


async def _list_datasets(show_json: bool) -> None:
    """List available datasets from the Semantic Scholar API."""
    async with aiohttp.ClientSession() as session:
        # Get latest release ID
        async with session.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        ) as releases_response:
            releases_response.raise_for_status()
            releases = await releases_response.json()

        release_id = releases["release_id"]
        print(f"Latest release ID: {release_id}")
        endpoint = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}"

        # Get datasets for the latest release
        async with session.get(endpoint) as datasets_response:
            datasets_response.raise_for_status()
            data = await datasets_response.json()

    if show_json:
        print(json.dumps(data["datasets"], indent=2))
    else:
        for dataset in data["datasets"]:
            print(dataset["name"], dataset["description"].strip(), sep="\n", end="\n\n")


@app.command(name="download", help="Download a dataset.", no_args_is_help=True)
def download_dataset(
    dataset_name: Annotated[
        str, typer.Argument(help="Name of the dataset to download.")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded files.")
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit the number of files to download. Useful for testing.",
        ),
    ] = None,
) -> None:
    """Download dataset files from the Semantic Scholar API to the output path.

    Prevents the user from accidentally exiting the script with Ctrl+C. Instead, it asks
    for confirmation before exiting.
    """
    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    output_path.mkdir(parents=True, exist_ok=True)

    arun_safe(_download, dataset_name, output_path, api_key, limit)


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
        write_file_bytes(output_path / "dataset.json", dataset)

        if "files" not in dataset or not dataset["files"]:
            die("No files found.")

        output_path.mkdir(exist_ok=True)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        tasks: list[Coroutine[None, None, None]] = []

        for url in dataset["files"][:limit]:
            file_name = urllib.parse.urlparse(str(url)).path.split("/")[-1]
            file_path = output_path / file_name

            tasks.append(_download_file(url, file_path, session, semaphore))

        await progress.gather(tasks, desc="Overall progress")


async def _download_file(
    url: str, path: Path, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> None:
    """Download a file from the given URL with a human-readable progress bar.

    The file is first downloaded to a .part file and then renamed. This function is a
    wrapper around _download_file that handles retries and errors.
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

        async with AsyncFile(part_path) as file:
            with tqdm(
                desc=str(display_path),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                async for chunk in response.content.iter_chunked(1024):
                    size = await file.write(chunk)
                    progress_bar.update(size)


class AsyncFile:
    """Async wrapper around writing bytes to a file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.loop = asyncio.get_event_loop()

    async def __aenter__(self) -> Self:
        """Open file in an executor."""

        def _open() -> io.BufferedWriter:
            return open(self.path, "wb")

        self.file = await self.loop.run_in_executor(None, _open)
        return self

    async def __aexit__(self, *_: object) -> bool | None:
        """Close file in an executor."""
        assert self.file
        await self.loop.run_in_executor(None, self.file.close)

    async def write(self, data: bytes) -> int:
        """Write to file in an executor."""
        assert self.file
        return await self.loop.run_in_executor(None, self.file.write, data)


if __name__ == "__main__":
    app()
