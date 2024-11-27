"""Tools for serialisation and deserialisation of Pydantic objects."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

type JSONPrimitive = str | bool | int | float
type JSONArray = Sequence[JSONValue]
type JSONObject = dict[str, JSONValue]
type JSONValue = JSONObject | JSONArray | JSONPrimitive


class Record(BaseModel, ABC):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @property
    @abstractmethod
    def id(self) -> str: ...


def load_data[T: BaseModel](
    path: Path, type_: type[T], use_alias: bool = True
) -> list[T]:
    """Load a list of data from JSON file in `path`.

    Args:
        path: File to read the data from. Must be a list of objects.
        type_: Type of the objects in the list.
        use_alias: If True, read object keys by using the real field names, not aliases.

    Returns:
        List of data with the `type_` format.
    """
    return TypeAdapter(
        list[type_], config=ConfigDict(populate_by_name=not use_alias)
    ).validate_json(path.read_bytes())


def save_data[T: BaseModel](
    path: Path, data: Sequence[T] | T, use_alias: bool = True
) -> None:
    """Save data in JSON file at `path`. Can be a single Pydantic object or a Sequence.

    Args:
        path: File where data will be saved. Creates its parent directory if it doesn't
            exist.
        data: The data to be saved. If it's a sequence, it must be non-empty.
        use_alias: If True, the output object keys will use the field alias, not the
            actual field name.
    """
    if not data:
        raise ValueError("Cannot save empty data")

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, Sequence):
        type_ = type(data[0])
        path.write_bytes(
            TypeAdapter(Sequence[type_]).dump_json(data, indent=2, by_alias=use_alias)
        )
    else:
        path.write_text(data.model_dump_json(indent=2, by_alias=use_alias))
