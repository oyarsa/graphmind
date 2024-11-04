"""Create demonstrations for few-shot prompting.

Takes an even number of entries for positive and negative approval decisions, finds
the review with lowest/highest reviews and uses it as the demonstration for the
rationale.

The input file is the output of the ASAP pipeline (asap_filtered.json).
The output is a file with the paper title, abstract, main text, approval decision and
the chosen rationale with its rating.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.gpt.model import Demonstration, DemonstrationType, Paper

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
        Path, typer.Argument(help="Input JSON with paper data (asap_filtered.json)")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file with the demonstrations")
    ],
    num_entries: Annotated[
        int, typer.Option("--entries", "-n", help="Number of entries for each type")
    ] = 10,
    seed: int = 0,
) -> None:
    random.seed(seed)

    papers = TypeAdapter(list[Paper]).validate_json(input_file.read_bytes())

    papers_positive = random.sample([p for p in papers if p.approval], num_entries)
    papers_negative = random.sample([p for p in papers if not p.approval], num_entries)

    demonstrations = [
        new_demonstration(paper, DemonstrationType.POSITIVE)
        for paper in papers_positive
    ] + [
        new_demonstration(paper, DemonstrationType.NEGATIVE)
        for paper in papers_negative
    ]
    output_file.write_bytes(
        TypeAdapter(list[Demonstration]).dump_json(demonstrations, indent=2)
    )


def new_demonstration(paper: Paper, type_: DemonstrationType) -> Demonstration:
    chosen_func = min if type_ is DemonstrationType.NEGATIVE else max
    chosen = chosen_func(paper.reviews, key=lambda x: x.rating)

    return Demonstration(
        title=paper.title,
        abstract=paper.abstract,
        text=paper.main_text(),
        approval=paper.approval,
        rationale=chosen.rationale,
        rating=chosen.rating,
        type=type_,
    )


if __name__ == "__main__":
    app()
