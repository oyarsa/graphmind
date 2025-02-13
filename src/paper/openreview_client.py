"""Fetch conference paper data from the OpenReview API."""

# pyright: basic
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from openreview import api

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview data.")
    ],
    venue_id: Annotated[
        str, typer.Option(help="Venue ID to fetch data.")
    ] = "ICLR.cc/2024/Conference",
) -> None:
    """Download all notes for a given conference."""
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
