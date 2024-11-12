"""Download paper information from the CrossRef API.

Takes the processed output from the ASAP pipeline as input (asap_filtered.json), finds
the unique reference titles and queries them on the Crossref API. This querying is done
by title, where we get the top 10 matches, of which we find the best matching by fuzzy
matching. There's a minimum fuzzy threshold (30 by default).

NB: Crossref has very poor performance, so we focus on using
external_data.semantic_scholar (Semantic Scholar API) instead. This script is likely to
break in the future, as it won't be maintained.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import habanero  # type: ignore
import typer
from tqdm import tqdm

from paper.util import fuzzy_ratio, run_safe


class CrossrefClient:
    def __init__(self, mailto: str | None = None) -> None:
        self.cr = habanero.Crossref(mailto=mailto or "")  # type: ignore

    def get_papers(self, title: str, fields: list[str], limit: int) -> dict[str, Any]:
        """Fetch top N papers using the title. Defaults to N=10."""
        return self.cr.works(  # type: ignore
            query_title=title, select=fields, limit=limit
        )


def get_best_paper(
    title: str, papers: Iterable[dict[str, Any]], fuzz_threshold: int
) -> dict[str, Any] | None:
    """Find the best paper by title fuzzy ratio. Returns None if there are no matches.

    Some paper entries don't have a title, and are obviously ignored. Also, we need
    abstracts, so even if a paper matches, we ignore it if it doesn't have an abstract.
    """
    best_paper = None
    best_ratio = fuzz_threshold

    for paper in papers:
        if (
            (paper_titles := paper["title"])
            and paper_titles
            and (paper_title := paper_titles[0].strip())
            and paper.get("abstract", "").strip()
        ):
            ratio = fuzzy_ratio(title, paper_title)
            if ratio > best_ratio:
                best_paper = paper
                best_ratio = ratio

    if best_paper:
        return {"fuzz_ratio": best_ratio} | best_paper
    else:
        return None


def download(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    fuzz_threshold: int,
    mailto: str | None,
    paper_limit: int,
) -> None:
    cr = CrossrefClient(mailto=mailto)  # type: ignore

    papers = json.loads(input_file.read_text())
    unique_titles: set[str] = {
        reference["title"] for paper in papers for reference in paper["references"]
    }
    fields = [f for field in fields_str.split(",") if (f := field.lower().strip())]

    output_full: list[dict[str, Any]] = []
    output_best: list[dict[str, Any]] = []

    for title in tqdm(unique_titles):
        meta = {"query": {"title": title}}
        result = cr.get_papers(title, fields, paper_limit)
        output_full.append(meta | result)

        if (papers := result.get("message", {}).get("items")) and (
            best := get_best_paper(title, papers, fuzz_threshold)
        ):
            output_best.append(meta | best)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "crossref_full.json").write_text(json.dumps(output_full))
    (output_path / "crossref_best.json").write_text(json.dumps(output_best))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path, typer.Argument(help="Input file (asap_filtered.json).")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Directory to save the downloaded information.")
    ],
    fields: Annotated[
        str, typer.Option(help="Comma-separated list of fields to retrieve.")
    ] = "title,author,published,abstract",
    ratio: Annotated[int, typer.Option(help="Minimum ratio for fuzzy matching")] = 30,
    mailto: Annotated[
        str | None,
        typer.Option(
            help="Email to add request header. Increases reputation and rate limit."
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option(help="Maximum number of papers returned for each query")
    ] = 10,
) -> None:
    run_safe(download, input_file, fields, output_path, ratio, mailto, limit)


if __name__ == "__main__":
    app()
