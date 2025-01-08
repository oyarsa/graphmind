"""Tools for serialisation and deserialisation of Pydantic objects."""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

from pydantic import BaseModel, ConfigDict, TypeAdapter

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
        """Unique identifcation for the object."""


@overload
def load_data[T: BaseModel](
    file: Path, type_: type[T], use_alias: bool = True, *, single: Literal[True]
) -> T: ...


@overload
def load_data[T: BaseModel](
    file: Path, type_: type[T], use_alias: bool = True, single: Literal[False] = False
) -> list[T]: ...


def load_data[T: BaseModel](
    file: Path,
    type_: type[T],
    use_alias: bool = True,
    single: bool = False,
) -> list[T] | T:
    """Load data from the JSON `file`: a list of objects or a single one if `single=True`.

    Args:
        file: File to read the data from. Must be a list of objects.
        type_: Type of the objects in the list.
        use_alias: If True, read object keys by using the real field names, not aliases.
        single: If True, return a single object. If False, return a list of objects.

    Returns:
        List of data with the `type_` format.
    """
    content = file.read_bytes()
    if single:
        return type_.model_validate_json(content)

    return TypeAdapter(
        list[type_], config=ConfigDict(populate_by_name=not use_alias)
    ).validate_json(content)


def save_data[T: BaseModel](
    file: Path, data: Sequence[T] | T, use_alias: bool = True
) -> None:
    """Save data in JSON `file`. Can be a single Pydantic object or a Sequence.

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
    if isinstance(data, Sequence):
        type_ = type(data[0])
        file.write_bytes(
            TypeAdapter(Sequence[type_]).dump_json(data, indent=2, by_alias=use_alias)
        )
    else:
        file.write_text(data.model_dump_json(indent=2, by_alias=use_alias))


def safe_load_json(file_path: Path) -> Any:
    """Load a JSON file, removing invalid UTF-8 characters."""
    return json.loads(file_path.read_text(encoding="utf-8", errors="replace"))


def replace_fields[T: BaseModel](obj: T, /, **kwargs: Any) -> T:
    """Return a new Pydantic object replacing specified fields with new values."""
    return obj.model_validate(obj.model_dump() | kwargs)
