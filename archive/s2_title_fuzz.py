"""Calculate citation quality for the S2 API results through fuzzy matching scores.

Takes as input the best matches from `paper_hypergraph.s2orc.retrieve_papers_semantic_scholar`
and calculates the fuzzy ratio between the retrieved best paper title and the original
title query (`title` vs `title_query`).

The saved result is identical to the input with an extra `fuzz_ratio` field.

Also takes a parameter "min_fuzzy" that if present, creates an additional file with only
entries with `fuzz_ratio` above that threshold. This file has the same name as the
normal output file, but with `.filtered.json` as extension.

This isn't necessary for Crossref because it already does the fuzzy matching there, so
the output already contains the `fuzzy_ratio` field. New versions of the Semantic Scholar
retrieval script also do this.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from paper_hypergraph.util import fuzzy_ratio

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to input JSON file (semantic_scholar_best.json)."),
    ],
    output_file: Annotated[Path, typer.Argument(help="Path to output JSON file.")],
    min_fuzzy: Annotated[
        int | None, typer.Option(help="Minimum fuzzy ratio to filter")
    ],
) -> None:
    input_data = json.loads(input_file.read_text())
    output_data = [
        paper | {"fuzz_ratio": fuzzy_ratio(paper["title_query"], paper["title"])}
        for paper in input_data
    ]
    output_file.write_text(json.dumps(output_data))

    if min_fuzzy:
        filtered_data = [
            paper for paper in output_data if paper["fuzz_ratio"] >= min_fuzzy
        ]
        filtered_file = output_file.with_name(output_file.stem + ".filtered.json")
        filtered_file.write_text(json.dumps(filtered_data, indent=2))


if __name__ == "__main__":
    app()
