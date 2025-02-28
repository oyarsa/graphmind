"""Fetch conference paper data from the OpenReview API and download LaTeX from arXiv.

Requires the following environment variables to be set:
- OPENREVIEW_USERNAME
- OPENREVIEW_PASSWORD

These are the standard credentials you use to log into the OpenReview website.
Example venue IDs:
- ICLR.cc/2024/Conference
- ICLR.cc/2025/Conference
- NeurIPS.cc/2024/Conference
"""

# pyright: basic
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import arxiv  # type: ignore
import backoff
import requests
import typer
from openreview import api
from tqdm import tqdm

from paper.util.cli import die

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
def latex(
    reviews_file: Annotated[
        Path,
        typer.Option(
            "--papers",
            help="Path to paper data from OpenReview. Can be a text file with one title"
            " per line, or the actual JSON from the `reviews` subcommand.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", help="Output directory for arXiv LaTeX source files."),
    ],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
    skip_file: Annotated[
        Path | None,
        typer.Option("--skip", help="File with paper titles to skip, one per line."),
    ] = None,
    batch_size: Annotated[
        int, typer.Option(help="Batch size to query the arXiv API.")
    ] = 50,
) -> None:
    """Download LaTeX source files from arXiv for OpenReview submissions.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean` and `--skip`.
    """
    if reviews_file.suffix == ".json":
        papers = json.loads(reviews_file.read_text())[:max_papers]
        titles: list[str] = [
            title
            for paper in papers
            if (title := paper.get("content", {}).get("title", {}).get("value"))
        ]
    else:
        titles = [
            title
            for line in reviews_file.read_text().splitlines()[:max_papers]
            if (title := line.strip())
        ]

    if not titles:
        die("No valid titles found.")
    print(f"Found {len(titles)} papers in input file")

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
            path.stem for path in output_dir.glob("*.tar.gz") if path.is_file()
        }

    arxiv_results = _get_arxiv(titles, batch_size)
    print(f"Found {len(arxiv_results)} papers on arXiv")

    downloaded_n = 0
    skipped_n = 0
    failed_n = 0
    output_dir.mkdir(exist_ok=True, parents=True)

    for result in tqdm(arxiv_results, desc="Downloading LaTeX sources"):
        if result.title in downloaded_prev:
            skipped_n += 1
            continue

        try:
            data = _download_latex_source(result.id)
            (output_dir / f"{result.title}.tar.gz").write_bytes(data)
            downloaded_n += 1
        except Exception as e:
            print(f"Error downloading LaTeX source for {result.title} - {type(e)}: {e}")
            failed_n += 1

    print(f"Downloaded : {downloaded_n}")
    print(f"Skipped    : {skipped_n}")
    print(f"Failed     : {failed_n}")


@dataclass(frozen=True, kw_only=True)
class ArxivResult:
    """Result of querying the arXiv API with a paper title from OpenReview."""

    title: str
    id: str


def _get_arxiv(paper_titles: list[str], batch_size: int) -> list[ArxivResult]:
    """Get arXiv information for the papers that are present there."""
    arxiv_client = arxiv.Client()

    arxiv_results: list[ArxivResult] = []
    for title_batch in tqdm(
        list(itertools.batched(paper_titles, batch_size)), desc="Querying arXiv"
    ):
        arxiv_results.extend(_batch_search_arxiv(arxiv_client, title_batch))

    return arxiv_results


def _batch_search_arxiv(
    client: arxiv.Client, titles: Sequence[str]
) -> list[ArxivResult]:
    """Search multiple titles at once on arXiv and return matching results."""
    or_queries = " OR ".join(f'ti:"{title}"' for title in titles)
    query = f"({or_queries})"
    results_map: dict[str, ArxivResult] = {}

    try:
        for result in client.results(
            arxiv.Search(query=query, max_results=len(titles))
        ):
            result_title = result.title.lower()
            for original_title in titles:
                if _similar_titles(original_title, result_title):
                    results_map[original_title.lower()] = ArxivResult(
                        id=result.entry_id.split("/")[-1],
                        title=result.title,
                    )
                    break
    except Exception as e:
        print(f"Error during batch search on arXiv: {e}")

    return [result for title in titles if (result := results_map.get(title.lower()))]


def _similar_titles(title1: str, title2: str) -> bool:
    """Check if two titles are similar (case-insensitive, stripped)."""
    t1 = title1.casefold().strip()
    t2 = title2.casefold().strip()
    return t1 == t2 or t1 in t2 or t2 in t1


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def _download_latex_source(arxiv_id: str) -> bytes:
    """Download LaTeX source (tar.gz) from arXiv for the given arXiv ID."""
    url = f"https://arxiv.org/src/{arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.content


@app.command(no_args_is_help=True)
def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    venue_id: Annotated[
        str, typer.Option(help="Venue ID to fetch data.")
    ] = ICLR_2024_ID,
) -> None:
    """Download all reviews and metadata for papers from a conference in OpenReview."""
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
    """Convert OpenReview API `Note` to dict with additional `details` object."""
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
    """Check if the review has the rating with given `name`.

    Checks whether the `content.{name}` field is non-empty.
    """
    return bool(review["content"].get(name))


def _has_rating(paper: dict[str, Any], name: str) -> bool:
    """Check if any review in `paper` has the rating with given `name`."""
    return any(_review_has_rating(r, name) for r in paper["details"]["replies"])


def _has_field(paper: dict[str, Any], name: str) -> bool:
    """Check if the `paper` has a field with `name` and non-empty value."""
    value = paper["content"].get(name, {}).get("value")
    if isinstance(value, str):
        value = value.strip()
    return bool(value)


if __name__ == "__main__":
    app()
