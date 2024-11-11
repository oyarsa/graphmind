"""Convert JSON files between snake_case and camelCase keys (can do both ways).

There's some inconsistency over time about whether our code outputs data with or without
Pydantic aliases. The data types use camel_case, but since inputs use camelCase, we
defined aliases.

My current thinking is that files that save external inupt should use the same keys, but
that's not the case for everything, so we might have to convert.
"""

import json
from collections.abc import Callable
from pathlib import Path

from paper.util import HelpOnErrorArgumentParser, JSONArray, JSONObject, JSONValue, die


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file")
    parser.add_argument(
        "output_file", type=Path, help="Path where the transformed JSON will be saved"
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        help="How to transform the keys",
        required=True,
    )

    args = parser.parse_args()
    convert_file(args.input_file, args.output_file, args.mode)


def convert_snake_to_camel_case(snake_str: str) -> str:
    """Convert a snake_case string to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def convert_camel_to_snake_case(camel_case: str) -> str:
    """Convert a camelCase string to snake_case."""
    return "".join(
        f"_{char.lower()}" if char.isupper() else char for char in camel_case
    ).lstrip("_")


VALID_MODES = {
    "s2c": convert_snake_to_camel_case,
    "c2s": convert_camel_to_snake_case,
}


def convert_file(input_path: Path, output_path: Path, mode: str) -> None:
    """Convert a JSON file with snake_case keys to camelCase.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path where the transformed JSON will be saved.
        mode: Key transformation mode. See `VALID_MODES`.
    """
    if mode not in VALID_MODES:
        die(f"Invalid transformation mode: '{mode}'. Must be one of {VALID_MODES}.")
    transform = VALID_MODES[mode]

    try:
        data: JSONArray = json.loads(input_path.read_text())
        transformed_data = transform_array(data, transform)
        output_path.write_text(json.dumps(transformed_data, indent=2))

        print(f"Successfully converted '{input_path}' to '{output_path}'.")
    except FileNotFoundError:
        die(f"File '{input_path}' was not found.")
    except json.JSONDecodeError as e:
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
        Structure where all nested object keys are tranformed.
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
    main()
