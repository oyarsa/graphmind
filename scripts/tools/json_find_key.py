"""Search JSON files for objects with matching keys."""
# pyright: strict

import contextlib
from pathlib import Path
from typing import Annotated

import orjson
import typer

from paper.util.serde import JSONValue


def search_object(obj: JSONValue, keyword: str, current_path: str = "") -> list[str]:
    """Recursively search the JSON object for a key matching key word.

    Assumes `keyword` is normalised. Normalises the object keys before comparison.
    """
    results: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{current_path}.{key}" if current_path else key
            if keyword in _normalise_key(key):
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
    keyword: Annotated[
        str, typer.Option("--keyword", "-k", help="Keyword to search for in JSON keys")
    ],
    paths: Annotated[
        list[Path] | None,
        typer.Option("--paths", "-p", help="List of paths to search for JSON files"),
    ] = None,
    show_matches: Annotated[
        bool,
        typer.Option("--matches", "-m", help="Print each match, not just the paths."),
    ] = False,
) -> None:
    """Find key matches to the keyword in the JSON files in `path`."""
    keyword = _normalise_key(keyword)
    paths = paths or [Path()]

    for path in paths or [Path()]:
        for file_path in path.rglob("*.json"):
            with contextlib.suppress(Exception):
                data = orjson.loads(file_path.read_text())
                matches = search_object(data, keyword)
                if not matches:
                    continue

                print(file_path)
                if show_matches:
                    for match in matches:
                        print(f"  - {match}")


def _normalise_key(key: str) -> str:
    return key.strip().casefold().replace(" ", "")


if __name__ == "__main__":
    app()
