"""Stream S2ORC datasets to AWS S3, or upload already-downloaded data.

If a path to local files is given, they will be uploaded. If not, we stream data from
the external dataset. Streaming means the contents are sent directly to S3 instead of
being fully downloaded locally.

Requires the following environment variables to be set:

- SEMANTIC_SCHOLAR_API_KEY
- S3_BUCKET_NAME
- AWS_REGION
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY

AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY can be obtained from the AWS IAM console.
The key should have permissions to create S3 buckets and upload to them.

SEMANTIC_SCHOLAR_API_KEY can be obtained at https://www.semanticscholar.org/product/api.

S3_BUCKET_NAME and AWS_REGION are up to you. For reference, the London AWS region is
'eu-west-2'.

The environment variables can be set on `.env` file.
"""

import asyncio
from collections.abc import Awaitable, Coroutine
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast
from urllib.parse import urlparse

import aioboto3
import aiohttp
import typer
from dotenv import load_dotenv
from types_aiobotocore_s3.client import S3Client

from paper.util import arun_safe, mustenv, progress

MAX_CONCURRENT_DOWNLOADS = 10
DOWNLOAD_TIMEOUT = 3600  # 1 hour timeout for each file
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds.

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    dataset_name: Annotated[
        str,
        typer.Argument(
            help="Dataset name (e.g., papers, abstracts, citations, tldrs)."
        ),
    ],
    local_path: Annotated[
        Path | None,
        typer.Option(
            "--local",
            help="Local path containing dataset files to upload instead of downloading.",
            exists=True,
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
    max_files: Annotated[
        int | None,
        typer.Option("--max-files", "-n", help="Maximum number of files to process."),
    ] = None,
) -> None:
    arun_safe(_main, dataset_name, local_path, max_files)


async def _main(
    dataset_name: str, local_path: Path | None, max_files: int | None
) -> None:
    load_dotenv()

    env = mustenv(
        "SEMANTIC_SCHOLAR_API_KEY",
        "S3_BUCKET_NAME",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    semantic_api_key = env["SEMANTIC_SCHOLAR_API_KEY"]
    bucket_name = env["S3_BUCKET_NAME"]
    aws_region = env["AWS_REGION"]
    aws_access_key_id = env["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    session = aioboto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    # The client is typed correctly, but pyright doesn't like it. Everything else works.
    async with session.client("s3") as s3:  # type: ignore
        if local_path:
            await _upload_local(s3, bucket_name, local_path, dataset_name, max_files)
        else:
            await _stream_files(
                s3, bucket_name, dataset_name, semantic_api_key, max_files
            )


async def _stream_files(
    s3: S3Client,
    bucket_name: str,
    dataset_name: str,
    api_key: str,
    max_files: int | None,
) -> None:
    """Download dataset files and upload them to S3 concurrently."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT), raise_for_status=True
    ) as session:
        # Get latest release's ID
        async with session.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        ) as response:
            release_id = (await response.json())["release_id"]
        typer.echo(f"Latest release ID: {release_id}")

        # Get the download links for the dataset
        async with session.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}/",
            headers={"x-api-key": api_key},
        ) as response:
            dataset = await response.json()

        if "files" not in dataset or not dataset["files"]:
            raise ValueError("No files found in dataset")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        tasks: list[Awaitable[None]] = []

        for url in dataset["files"][:max_files]:
            filename = Path(urlparse(str(url)).path).name
            if not filename:
                typer.echo(f"Invalid url from dataset: {url}. Skipping.", err=True)
                continue

            object_name = f"{dataset_name}/{filename}"
            tasks.append(
                _stream_file(session, url, s3, bucket_name, object_name, semaphore)
            )

        await progress.gather(tasks, desc="Streaming files")


async def _stream_file(
    session: aiohttp.ClientSession,
    url: str,
    s3: S3Client,
    bucket_name: str,
    object_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Stream file from S2ORC to S3, retrying errors with exponential backoff."""
    delay = RETRY_BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            await _try_stream_file(
                session, url, s3, bucket_name, object_name, semaphore
            )
        except Exception as e:
            typer.echo(
                f"Error processing {object_name}: {e}."
                f" Retrying after {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})",
                err=True,
            )
            await asyncio.sleep(delay)
            delay *= RETRY_BACKOFF_BASE
        else:
            return

    typer.echo(
        f"Failed to process {object_name} after {MAX_RETRIES} attempts.", err=True
    )


async def _try_stream_file(
    session: aiohttp.ClientSession,
    url: str,
    s3: S3Client,
    bucket_name: str,
    object_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Try to stream the file from S2ORC to S3.

    Raises:
        aiohttp.ClientError: if an error happens with the HTTP request.
        botocore.exceptions.ClientError: if an error happens with the S3 upload.
    """

    async with semaphore, session.get(url) as response:
        total_size = int(response.headers.get("content-length", 0))

        with progress.filebar(
            desc=f"Streaming {object_name[-16:]}", size=total_size
        ) as progress_bar:
            stream = cast(BinaryIO, response.content)
            await s3.upload_fileobj(
                stream, bucket_name, object_name, Callback=progress_bar.update
            )


async def _upload_local(
    s3: S3Client,
    bucket_name: str,
    local_path: Path,
    dataset_name: str,
    max_files: int | None,
) -> None:
    """Upload local files to S3 concurrently."""
    files = list(local_path.glob("*.gz"))
    typer.echo(f"Found {len(files)} files to upload in {local_path}")

    tasks: list[Coroutine[Any, Any, None]] = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    for file_path in files[:max_files]:
        object_name = f"{dataset_name}/{file_path.name}"
        tasks.append(
            _upload_file_to_s3(s3, bucket_name, file_path, object_name, semaphore)
        )

    await progress.gather(tasks, desc="Uploading local files")


async def _upload_file_to_s3(
    s3: S3Client,
    bucket_name: str,
    file_path: Path,
    object_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Upload a single file to S3, retrying errors with exponential backoff."""
    delay = RETRY_BACKOFF_BASE

    for attempt in range(MAX_RETRIES):
        try:
            await _try_upload_file_to_s3(
                s3, bucket_name, file_path, object_name, semaphore
            )
        except Exception as e:
            typer.echo(
                f"Error uploading {object_name}: {e}."
                f" Retrying after {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})",
                err=True,
            )
            await asyncio.sleep(delay)
            delay *= RETRY_BACKOFF_BASE
        else:
            typer.echo(f"Successfully uploaded {object_name}")
            return

    typer.echo(
        f"Failed to upload {object_name} after {MAX_RETRIES} attempts.", err=True
    )


async def _try_upload_file_to_s3(
    s3: S3Client,
    bucket_name: str,
    file_path: Path,
    object_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Try to upload a single file to S3.

    Raises:
        botocore.exceptions.ClientError: if an error happens with the S3 upload.
    """

    async with semaphore:
        with progress.filebar(
            desc=object_name[-16:], size=file_path.stat().st_size
        ) as progress_bar:
            await s3.upload_file(
                str(file_path), bucket_name, object_name, Callback=progress_bar.update
            )


if __name__ == "__main__":
    app()
