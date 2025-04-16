"""Tools for serialisation and deserialisation of Pydantic objects."""

import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypeGuard,
    cast,
    get_origin,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

type JSONPrimitive = str | bool | int | float
type JSONArray = Sequence[JSONValue]
type JSONObject = dict[str, JSONValue]
type JSONValue = JSONObject | JSONArray | JSONPrimitive


class Record(BaseModel, ABC):
    """Immutable model type with a unique ID."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identification for the object."""


def save_data_jsonl(
    file: Path, data: BaseModel | Sequence[BaseModel], mode: Literal["w", "a"] = "a"
) -> None:
    """Save Pydantic object as a new line in `file`.

    Args:
        file: File where data will be saved. Creates its parent directory if it doesn't
            exist.
        data: The data to be saved.
        mode: How to open the file. Defaults to append.
    """
    if isinstance(data, BaseModel):
        data = [data]

    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open(mode) as f:
        for entry in data:
            f.write(entry.model_dump_json() + "\n")


def load_data_jsonl[T: BaseModel](file: Path, type_: type[T]) -> list[T]:
    """Load data from a JSON Lines (JSONL) `file`: a collection of objects, one per line.

    Each line in the file should be a valid JSON object.

    Args:
        file: File path to read the data from, bytes content, or string content.
        type_: Type of the objects to parse.

    Returns:
        List of data with the `type_` format, or a single object if `single=True`.

    Raises:
        `ValidationError` if the data is incompatible with `type_`.
        `OSError` if the file operations fail.
        `ValueError` if no valid objects are found in the file.
    """

    try:
        result: list[T] = []
        errors: list[str] = []

        with file.open() as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    result.append(type_.model_validate_json(line))
                except ValidationError as e:
                    # Collect errors with line numbers for better debugging
                    errors.append(f"Line {i}: {e}")

        if errors and not result:
            # If all lines failed validation, raise an error
            raise ValidationError(  # noqa: TRY301
                f"All lines in {file} failed validation for {get_full_type_name(type_)}. "
                f"First few errors: {'; '.join(errors[:3])}"
            )

    except ValidationError as e:
        raise ValidationError(
            f"Data from {file} is not valid for {get_full_type_name(type_)}"
        ) from e
    except json.JSONDecodeError as e:
        line_num = e.lineno if hasattr(e, "lineno") else "unknown"
        raise ValueError(f"Invalid JSON at line {line_num} in {file}: {e}") from e
    else:
        return result


@overload
def load_data[T: BaseModel](
    file: Path | bytes, type_: type[T], use_alias: bool = True, *, single: Literal[True]
) -> T: ...


@overload
def load_data[T: BaseModel](
    file: Path | bytes,
    type_: type[T],
    use_alias: bool = True,
    single: Literal[False] = False,
) -> list[T]: ...


# TODO: Improve validation error reporting. It currently prints the errors for _every_
# item in the list, which is not helpful.
def load_data[T: BaseModel](
    file: Path | bytes,
    type_: type[T],
    use_alias: bool = True,
    single: bool = False,
) -> list[T] | T:
    """Load data from the JSON `file`: a list of objects or a single one if `single=True`.

    Args:
        file: File to read the data from, or the actual data in bytes form.
        type_: Type of the objects in the list.
        use_alias: If True, read object keys by using the real field names, not aliases.
        single: If True, return a single object. If False, return a list of objects.

    Returns:
        List of data with the `type_` format.

    Raises:
        `ValidationError` if file already exists and its data is incompatible with
        `type_`.
        `OSError` if the file operations fail.
    """
    if isinstance(file, Path):
        content = file.read_bytes()
    else:
        content = file

    try:
        if single:
            return type_.model_validate_json(content)

        return TypeAdapter(
            list[type_], config=ConfigDict(populate_by_name=not use_alias)
        ).validate_json(content)
    except ValidationError as e:
        source = file if isinstance(file, Path) else "bytes"
        raise ValidationError(
            f"Data from {source} is not valid for {get_full_type_name(type_)}"
        ) from e


def save_data[T: BaseModel](
    file: Path, data: Sequence[T] | T | Any, use_alias: bool = True
) -> None:
    """Save data in JSON `file`. Can be a single Pydantic object or a Sequence, or Any.

    If `data` is a Pydantic object or Sequence of one, we'll use Pydantic to convert
    to JSON. If not, we'll use `json.dumps` directly.

    Args:
        file: File where data will be saved. Creates its parent directory if it doesn't
            exist.
        data: The data to be saved. If it's a sequence, it must be non-empty.
        use_alias: If True, the output object keys will use the field alias, not the
            actual field name.
    """
    if not data:
        raise ValueError("Cannot save empty data")

    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(_dump_data_to_json(data, use_alias=use_alias))


def _dump_data_to_json[T: BaseModel](
    data: Sequence[T] | T | Any, use_alias: bool
) -> str:
    """Return a JSON string representation of `data`.

    - If `data` is a single BaseModel, use its `.model_dump_json()`.
    - If `data` is a non-empty sequence of BaseModel, use `TypeAdapter`.
    - Otherwise, fall back to `json.dumps`.
    """

    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2, by_alias=use_alias)

    if isinstance(data, Sequence):
        data = cast(Sequence[Any], data)
        if _is_model_list(data):
            type_ = type(data[0])
            return TypeAdapter(Sequence[type_]).dump_json(data).decode()

    return json.dumps(data, indent=2)


def _is_model_list(val: Sequence[object]) -> TypeGuard[Sequence[BaseModel]]:
    """Determine whether all objects in the list are Pydantic models."""
    return all(isinstance(x, BaseModel) for x in val)


def get_full_type_name[T](type_: type[T]) -> str:
    """Get full name of type, including full module path."""
    # Handle generic types (List[str], etc.)
    origin = get_origin(type_)
    if origin is not None:
        type_ = origin

    # Try to find the original module
    for module_name, module in sys.modules.items():
        if hasattr(module, type_.__name__):
            obj = getattr(module, type_.__name__)
            if obj is type_:
                return f"{module_name}.{type_.__qualname__}"

    # Fallback to default
    return f"{type_.__module__}.{type_.__qualname__}"


def safe_load_json(file_path: Path) -> Any:
    """Load a JSON file, removing invalid UTF-8 characters."""
    return json.loads(file_path.read_text(encoding="utf-8", errors="replace"))


@runtime_checkable
class PydanticProtocol(Protocol):
    """Class with `model_dump` and `model_validate` functions.

    Should be compatible with Pydantic BaseModel classes.

    Used for the `replace_fields` function so we can use it in other protocols, as we
    can't inherit from abstract classes in protocols.
    """

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Dump BaseModel to a dictionary, including nested objects."""
        ...

    def model_validate(self, obj: Any, *args: Any, **kwargs: Any) -> Self:
        """Construct BaseModel object from dictionary."""
        ...


def replace_fields[T: PydanticProtocol](obj: T, /, **kwargs: Any) -> T:
    """Return a new Pydantic object replacing specified fields with new values."""
    return obj.model_validate(obj.model_dump() | kwargs)
