"""Remove papers whose ratings have too much variance.

This is determined by the difference between the minimum and maximum ratings. If it's
greater than 3, the paper is removed from the dataset.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.asap.model import Paper


def _keep_paper(paper: Paper) -> bool:
    """Paper is kept if the difference between min and max ratings is <= 3."""
    ratings = [r.rating for r in paper.reviews]
    return max(ratings) - min(ratings) <= 3


def filter_ratings(input_file: Path, output_file: Path) -> None:
    """Remove papers whose ratings have too much variance from the dataset.

    The input file is the output of `paper.asap.extract`. The output has
    the same format as the input.
    """
    data = TypeAdapter(list[Paper]).validate_json(input_file.read_bytes())
    output = [p for p in data if _keep_paper(p)]

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    output_file.write_bytes(TypeAdapter(list[Paper]).dump_json(output, indent=2))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input: Annotated[Path, typer.Argument(help="Path to input (extracted) JSON file.")],
    output: Annotated[Path, typer.Argument(help="Path to output filtered JSON file.")],
) -> None:
    """Remove papers whose ratings have too much variance."""
    filter_ratings(input, output)


if __name__ == "__main__":
    app()
