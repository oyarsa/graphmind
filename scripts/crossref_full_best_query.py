"""Add paper query information to Crossref "best" file.

The original version of the Crossref script only added the query stuff to the "full"
file, but it's interesting to have it in the "best" file too to compare the best output
with the original papers.

This is done by going through the "full" file and doing the same fuzzy comparison the
original script did.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm

from paper.external_data.crossref import get_best_paper
from paper.util import run_safe


def download(
    input_file: Path,
    output_file: Path,
    fuzz_threshold: int,
) -> None:
    input_papers = json.loads(input_file.read_text())

    output_best: list[dict[str, Any]] = []

    for paper_full in tqdm(input_papers):
        if (papers := paper_full.get("message", {}).get("items")) and (
            best := get_best_paper(paper_full["query"]["title"], papers, fuzz_threshold)
        ):
            output_best.append(
                {
                    "query": {
                        "title": paper_full["query"]["title"],
                        "author": paper_full["query"]["author"],
                    }
                }
                | best
            )

    print(f"Before: {len(input_papers)} papers")
    print(
        f"After : {len(output_best)} papers ({len(output_best) / len(input_papers):.2%})"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_best))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path, typer.Argument(help="File containing paper titles, one per line.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="File to save the downloaded information.")
    ],
    ratio: Annotated[int, typer.Option(help="Minimum ratio for fuzzy matching.")] = 30,
) -> None:
    run_safe(download, input_file, output_file, ratio)


if __name__ == "__main__":
    app()
