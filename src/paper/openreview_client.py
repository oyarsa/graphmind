"""Fetch conference paper data from the OpenReview API."""

# pyright: basic
import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import typer
from openreview import api

from paper.util import progress

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(no_args_is_help=True)
def pdfs(
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Path to downloaded OpenReview data.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to output directory for the PDFs."),
    ],
    max_concurrent: Annotated[
        int,
        typer.Option(
            "--max-concurrent", "-j", help="Maximum number of concurrent downloads"
        ),
    ] = 10,
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to download. If None, downloads all.",
        ),
    ] = None,
) -> None:
    """Download PDFs from OpenReview data."""
    asyncio.run(_pdfs(input_file, output_dir, max_concurrent, num_papers))


async def _pdfs(
    input_file: Path, output_dir: Path, max_concurrent: int, num_papers: int | None
) -> None:
    papers: list[dict[str, Any]] = json.loads(input_file.read_bytes())[:num_papers]
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_n = 0
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        tasks = [
            _download_to_file(session, semaphore, paper["content"]["pdf"]["value"])
            for paper in papers
        ]
        for paper, task in zip(papers, progress.as_completed(tasks)):
            name = paper["content"]["title"]["value"]
            try:
                result = await task
                (output_dir / f"{name}.pdf").write_bytes(result)
                downloaded_n += 1
            except Exception as e:
                print(f"Error downloading '{name}': {e}")

    print(f"Downloaded {downloaded_n} papers.")


async def _download_to_file(
    session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, path: str
) -> bytes:
    url = f"https://openreview.net{path}"
    async with semaphore, session.get(url) as response:
        return await response.read()


@app.command(no_args_is_help=True)
def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview data.")
    ],
    venue_id: Annotated[
        str, typer.Option(help="Venue ID to fetch data.")
    ] = "ICLR.cc/2024/Conference",
) -> None:
    """Download all reviews and metadata for papers from a given conference."""
    client = api.OpenReviewClient(baseurl="https://api2.openreview.net")

    venue_group = client.get_group(venue_id).content
    if venue_group is None:
        print(f"Invalid venue ID: '{venue_id}'")
        sys.exit(1)

    submission_name = venue_group["submission_name"]["value"]
    submissions_raw = client.get_all_notes(
        invitation=f"{venue_id}/-/{submission_name}", details="replies"
    )
    if not submissions_raw:
        print("Empty submissions list")
        sys.exit(1)

    submissions_all = [_note_to_dict(s) for s in submissions_raw]
    submissions_valid = [s for s in submissions_all if _is_valid(s, "contribution")]

    print("Submissions - all:", len(submissions_all))
    print("Submissions - valid:", len(submissions_valid))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "openreview_all.json").write_text(json.dumps(submissions_all))
    (output_dir / "openreview_valid.json").write_text(json.dumps(submissions_valid))


def _note_to_dict(note: api.Note) -> dict[str, Any]:
    return note.to_json() | {"details": note.details}


def _is_valid(paper: dict[str, Any], rating: str) -> bool:
    """Check if paper has at least one review with `rating`, PDF, title and abstract."""
    return all((
        _has_rating(paper, rating),
        _has_field(paper, "pdf"),
        _has_field(paper, "title"),
        _has_field(paper, "abstract"),
    ))


def _review_has_rating(review: dict[str, Any], name: str) -> bool:
    return bool(review["content"].get(name))


def _has_rating(paper: dict[str, Any], name: str) -> bool:
    return any(_review_has_rating(r, name) for r in paper["details"]["replies"])


def _has_field(paper: dict[str, Any], name: str) -> bool:
    value = paper["content"].get(name, {}).get("value")
    if isinstance(value, str):
        value = value.strip()
    return bool(value)


if __name__ == "__main__":
    app()
