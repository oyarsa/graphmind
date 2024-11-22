"""Find all the unique titles of papers in a JSON file and save them to a text file.

The input JSON file should have the following structure (nested):
- List of objects, each with a "paper" key
- Each "paper" object has a "references" key with a list of objects
- Each "references" object has a "title" key with a string value
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, TypeAdapter


class Reference(BaseModel):
    title: str


class Paper(BaseModel):
    references: Sequence[Reference]


class Data(BaseModel):
    paper: Paper


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    infile: Annotated[Path, typer.Argument(help="Path to the input JSON file.")],
    outfile: Annotated[Path, typer.Argument(help="Path to the output JSON file.")],
) -> None:
    data = TypeAdapter(list[Data]).validate_json(infile.read_bytes())
    titles = {p.title.strip() for d in data for p in d.paper.references}

    print(f"Found {len(titles)} unique titles")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(sorted(titles)))


if __name__ == "__main__":
    app()
