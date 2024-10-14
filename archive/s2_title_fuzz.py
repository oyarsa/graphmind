"""Calculate citation quality for the S2 API results through fuzzy matching scores.

Takes as input the best matches from `paper_hypergraph.s2orc.retrieve_papers_semantic_scholar`
and calculates the fuzzy ratio between the retrieved best paper tiel and the original
title query.

Saves the result as a JSON file containing array where each object represents a paper
in the input file, with the following fields:
- title_query (str): the original paper title used to query the S2 API
- title_result (str): the title of the retrieved paper from the API
- fuzz_ratio (int): number from 0-100 from fuzzy matching

This isn't necessary for Crossref because it already does the fuzzy matching there, so
the output already contains the `fuzzy_ratio` field.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, TypeAdapter

from paper_hypergraph.util import fuzzy_ratio


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title_query: str
    title: str


class Output(BaseModel):
    title_query: str
    title_result: str
    fuzz_ratio: int


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
) -> None:
    input_data = TypeAdapter(list[Paper]).validate_json(input_file.read_text())

    output_data = [
        Output(
            title_query=p.title_query,
            title_result=p.title,
            fuzz_ratio=fuzzy_ratio(p.title_query, p.title),
        )
        for p in input_data
    ]

    output_file.write_bytes(TypeAdapter(list[Output]).dump_json(output_data))


if __name__ == "__main__":
    app()
