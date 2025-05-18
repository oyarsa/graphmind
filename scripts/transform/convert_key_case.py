"""Convert JSON files between snake_case and camelCase keys (can do both ways).

There's some inconsistency over time about whether our code outputs data with or without
Pydantic aliases. The data types use camel_case, but since inputs use camelCase, we
defined aliases.

My current thinking is that files that save external inupt should use the same keys, but
that's not the case for everything, so we might have to convert.
"""

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import orjson
import typer

from paper.util.cli import die
from paper.util.serde import JSONArray, JSONObject, JSONValue


class ConvertMode(StrEnum):
    """Convert from snake_case to CamelCase, or vice-versa."""

    SNAKE_TO_CAMEL = "snake_to_camel"
    CAMEL_TO_SNAKE = "camel_to_snake"

    def convert(self, text: str) -> str:
        """Convert `text` according to the mode."""
        match self:
            case self.SNAKE_TO_CAMEL:
                components = text.split("_")
                return components[0] + "".join(x.title() for x in components[1:])
            case self.CAMEL_TO_SNAKE:
                return "".join(
                    f"_{char.lower()}" if char.isupper() else char for char in text
                ).lstrip("_")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_path: Annotated[Path, typer.Argument(help="Path to the input JSON file.")],
    output_path: Annotated[
        Path, typer.Argument(help="Path where the transformed JSON will be saved.")
    ],
    mode: Annotated[
        ConvertMode,
        typer.Option(help="How to transform the keys."),
    ],
) -> None:
    """Convert a JSON file with snake_case keys to camelCase.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path where the transformed JSON will be saved.
        mode: Key transformation mode. See `VALID_MODES`.
    """
    try:
        data: JSONArray = orjson.loads(input_path.read_bytes())
        transformed_data = transform_array(data, mode.convert)
        output_path.write_bytes(orjson.dumps(transformed_data))

        print(f"Successfully converted '{input_path}' to '{output_path}'.")
    except FileNotFoundError:
        die(f"File '{input_path}' was not found.")
    except orjson.JSONDecodeError as e:
        die(f"File '{input_path}' contains invalid JSON: {e}")
    except Exception as e:
        die(f"An unexpected error occurred: {e}")


def transform_array(arr: JSONArray, transform: Callable[[str], str]) -> JSONArray:
    """Recursively transform all items in array using `transform`.

    Args:
        arr: Input array of JSON values (nested arrays, objects or primitives).
        transform: Function to transform object key (e.g. camelCase -> snake_case).

    Returns:
        Array where all nested object keys are transformed.
    """
    return [transform_keys(val, transform) for val in arr]


def transform_keys(obj: JSONValue, transform: Callable[[str], str]) -> JSONValue:
    """Recursively transform all object keys using `transform`.

    Args:
        obj: Input dictionary or list that may contain nested dictionaries/lists.
        transform: Function to transform object key (e.g. camelCase -> snake_case).

    Returns:
        Structure where all nested object keys are transformed.
    """
    if isinstance(obj, list):
        return [transform_keys(item, transform) for item in obj]

    if not isinstance(obj, dict):
        return obj

    return transform_object(obj, transform)


def transform_object(obj: JSONObject, transform: Callable[[str], str]) -> JSONObject:
    """Recursively transform all fields in object using `transform`.

    Args:
        obj: Input object, potentially nested.
        transform: Function to transform object key (e.g. camelCase -> snake_case).

    Returns:
        Object where all nested object keys are transformed.
    """
    transformed: JSONObject = {}
    for key, value in obj.items():
        camel_key = transform(key)

        # Recursively transform nested dictionaries and lists
        if isinstance(value, dict | list):
            transformed[camel_key] = transform_keys(value, transform)
        else:
            transformed[camel_key] = value

    return transformed


if __name__ == "__main__":
    app()
