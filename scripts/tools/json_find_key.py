"""Search JSON files for objects with matching keys."""
# pyright: strict

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from paper.util.serde import JSONValue


def search_object(obj: JSONValue, keyword: str, current_path: str = "") -> list[str]:
    results: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if keyword in key.lower().replace(" ", ""):
                results.append(new_path)
            results.extend(search_object(value, keyword, new_path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            new_path = f"{current_path}[{index}]"
            results.extend(search_object(item, keyword, new_path))

    return results


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    keyword: Annotated[str, typer.Argument(help="Keyword to search for in JSON keys")],
    paths: Annotated[
        Sequence[Path], typer.Argument(help="List of paths to search for JSON files")
    ] = (Path(),),
) -> None:
    keyword = keyword.lower().replace(" ", "")

    for path in paths:
        for file_path in path.rglob("*.json"):
            data = json.loads(file_path.read_text())

            matches = search_object(data, keyword)
            if not matches:
                continue

            print(file_path)
            for match in matches:
                print(f"  {match}")
            print()


if __name__ == "__main__":
    app()
