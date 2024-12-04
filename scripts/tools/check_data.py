"""Given a class and a file, check if the type is valid for the data.

We only check the first item for errors so we don't get thousands of complaints from
Pydantic. If that goes well, we load the whole thing just to be sure.
"""

import json
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from paper.util.serde import load_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[Path, typer.Argument(help="Path to JSON data.")],
    class_path: Annotated[
        str,
        typer.Argument(help="Name of the type to use. Must be a Pydantic BaseModel."),
    ],
) -> None:
    """Try to open data file using class."""
    module_path, type_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    type_: type[BaseModel] = getattr(module, type_name)

    raw = json.loads(input_file.read_bytes())
    first = type_.model_validate(raw[0])

    table = Table("Key", "Type")
    for key, val in first.model_dump().items():
        table.add_row(key, type(val).__qualname__)
    Console().print(table)

    parsed = load_data(input_file, type_)
    print(f"{len(parsed)} items.")


if __name__ == "__main__":
    app()
