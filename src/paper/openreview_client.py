"""Fetch conference paper data from the OpenReview API.

Requires the following environment variables to be set:
- OPENREVIEW_USERNAME
- OPENREVIEW_PASSWORD

These are the standard credentials you use to log into the OpenReview website.
"""

# pyright: basic
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import backoff
import typer
from openreview import api
from tqdm import tqdm

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

ICLR_2024_ID = "ICLR.cc/2024/Conference"


@app.command(no_args_is_help=True)
def pdfs(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview PDF files.")
    ],
    venue_id: Annotated[
        str, typer.Option(help="Venue ID to fetch data.")
    ] = ICLR_2024_ID,
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to download. If None, downloads all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
    skip_file: Annotated[
        Path | None,
        typer.Option("--skip", help="File with paper names to skip, one per line."),
    ] = None,
) -> None:
    """Download paper PDF attachments."""
    client = api.OpenReviewClient(baseurl="https://api2.openreview.net")
    notes = client.get_all_notes(invitation=f"{venue_id}/-/Submission")

    downloaded_prev: set[str]
    if clean_run:
        downloaded_prev = set()
    elif skip_file is not None:
        downloaded_prev = {
            name
            for line in skip_file.read_text().splitlines()
            if (name := line.strip())
        }
    else:
        downloaded_prev = {
            path.stem for path in output_dir.glob("*.pdf") if path.is_file()
        }

    output_dir.mkdir(exist_ok=True, parents=True)

    downloaded_n = 0
    skipped_n = 0
    failed_n = 0
    for note in tqdm(notes[:num_papers]):
        if note.content.get("pdf", {}).get("value"):
            name = note.content["title"]["value"]
            if name in downloaded_prev:
                skipped_n += 1
                continue

            try:
                data = _download_pdf(client, note.id)
                (output_dir / f"{name}.pdf").write_bytes(data)
                downloaded_n += 1
            except Exception as e:
                print(f"Error downloading paper '{name}' - {type(e)}: {e}")
                failed_n += 1

    print(f"Downloaded : {downloaded_n}")
    print(f"Skipped    : {skipped_n}")
    print(f"Failed     : {failed_n}")


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _download_pdf(client: api.OpenReviewClient, note_id: str) -> bytes:
    if data := client.get_attachment(note_id, "pdf"):
        return data
    raise ValueError("Empty attachment")


@app.command(no_args_is_help=True)
def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    venue_id: Annotated[
        str, typer.Option(help="Venue ID to fetch data.")
    ] = ICLR_2024_ID,
) -> None:
    """Download all reviews and metadata for papers from a given conference."""
    client = api.OpenReviewClient(baseurl="https://api2.openreview.net")
    submissions_raw = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission", details="replies"
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
