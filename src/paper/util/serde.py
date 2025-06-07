"""Tools for serialisation and deserialisation of Pydantic objects."""

import gzip
import io
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypeGuard,
    cast,
    get_origin,
    runtime_checkable,
)

import orjson
import zstandard as zstd
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

type JSONPrimitive = str | bool | int | float
type JSONArray = Sequence[JSONValue]
type JSONObject = dict[str, JSONValue]
type JSONValue = JSONObject | JSONArray | JSONPrimitive


class Compress(StrEnum):
    """Supported compression types."""

    GZIP = "gzip"
    ZSTD = "zstd"
    NONE = "none"

    @classmethod
    def infer(cls, file: Path) -> "Compress":
        """Infer compression method from file path."""
        match file.suffix.casefold():
            case ".zst":
                return Compress.ZSTD
            case ".gz":
                return Compress.GZIP
            case _:
                return Compress.NONE


@contextmanager
def open_compressed(
    file: Path, mode: Literal["r", "w", "a"], encoding: str = "utf-8"
) -> Generator[io.TextIOBase, None, None]:
    """Open a file with automatic compression detection based on extension."""
    match file.suffix.lower():
        case ".gz":
            # gzip supports text mode directly
            text_mode = mode + "t"
            with gzip.open(file, text_mode, encoding=encoding) as f:
                yield f

        case ".zst":
            # zstandard requires different handling for read/write
            if "r" in mode:
                dctx = zstd.ZstdDecompressor()
                with open(file, "rb") as fh, dctx.stream_reader(fh) as reader:
                    text_wrapper = io.TextIOWrapper(reader, encoding=encoding)
                    try:
                        yield text_wrapper
                    finally:
                        text_wrapper.detach()  # Prevent closing the underlying stream
            else:
                # For write/append modes
                cctx = zstd.ZstdCompressor()
                file_mode = "ab" if mode == "a" else "wb"
                with open(file, file_mode) as fh, cctx.stream_writer(fh) as compressor:
                    text_wrapper = io.TextIOWrapper(compressor, encoding=encoding)
                    try:
                        yield text_wrapper
                    finally:
                        text_wrapper.flush()
                        text_wrapper.detach()  # Prevent closing the underlying stream

        case _:
            # No compression
            with file.open(mode, encoding=encoding) as f:
                yield f


class Record(BaseModel, ABC):
    """Immutable model type with a unique ID."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identification for the object."""


class SerdeError(Exception):
    """Exceptions raised when loading/saving objects."""


def get_compressed_file_path(file: Path, compress: Compress | None) -> Path:
    """Determine the actual file path based on compression settings.

    Args:
        file: Original file path.
        compress: Compression type to use. Compress.NONE keeps the original path as-is.

    Returns:
        File path with appropriate extension for the compression type.
    """
    if compress is None:
        compress = Compress.infer(file)

    match compress:
        case Compress.NONE:
            return file
        case Compress.GZIP:
            suffix = ".gz"
        case Compress.ZSTD:
            suffix = ".zst"

    return (
        file if str(file).endswith(suffix) else file.with_suffix(file.suffix + suffix)
    )


def read_file_bytes(file: Path) -> bytes:
    """Read file contents, automatically detecting and decompressing if needed.

    Args:
        file: Path to file to read.

    Returns:
        Decompressed file contents as bytes.
    """
    match file.suffix:
        case ".zst":
            dctx = zstd.ZstdDecompressor()
            with file.open("rb") as f:
                return dctx.decompress(f.read())
        case ".gz":
            with gzip.open(file, "rb") as f:
                return f.read()
        case _:
            return file.read_bytes()


def write_file_bytes(file: Path, content: bytes, compress: Compress | None) -> None:
    """Write bytes to file, optionally compressing with zstandard or gzip.

    Args:
        file: Path to write to. Extension will be added based on compression type.
        content: Bytes to write.
        compress: Compression type to use. If None, will infer from the file path.
    """
    actual_file = get_compressed_file_path(file, compress)
    actual_file.parent.mkdir(parents=True, exist_ok=True)

    if compress is None:
        compress = Compress.infer(file)

    match compress:
        case Compress.GZIP:
            with gzip.open(actual_file, "wb") as f:
                f.write(content)
        case Compress.ZSTD:
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(content)
            actual_file.write_bytes(compressed)
        case Compress.NONE:
            actual_file.write_bytes(content)


def save_data_jsonl(
    file: Path,
    data: BaseModel | Sequence[BaseModel],
    mode: Literal["w", "a"] = "a",
    compress: Compress | None = None,
) -> None:
    """Save Pydantic object as a new line in `file`.

    If compress is provided, the file extension will be modified accordingly and
    compression applied. If not provided, compression is inferred from the file extension.

    Args:
        file: File where data will be saved. Creates its parent directory if it doesn't
            exist.
        data: The data to be saved.
        mode: How to open the file. Defaults to append.
        compress: Compression type to use. If provided, modifies file extension. If None,
            infers from the file path.
    """
    if isinstance(data, BaseModel):
        data = [data]

    actual_file = get_compressed_file_path(file, compress)
    actual_file.parent.mkdir(parents=True, exist_ok=True)

    with open_compressed(actual_file, mode) as f:
        for entry in data:
            f.write(entry.model_dump_json() + "\n")


def load_data_jsonl[T: BaseModel](file: Path, type_: type[T]) -> list[T]:
    """Load data from a JSON Lines (JSONL) `file`: a collection of objects, one per line.

    Each line in the (decompressed) file should be a valid JSON object.

    If the path ends with `.gz` or `.zst`, the data is decompressed with gzip or zstd,
    respectively.

    Args:
        file: File path to read the data from.
        type_: Type of the objects to parse.

    Returns:
        List of data with the `type_` format.

    Raises:
        `SerdeError` if the data is incompatible with `type_` or the JSON is invalid.
    """
    result: list[T] = []
    errors: list[str] = []

    try:
        with open_compressed(file, "r") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    result.append(type_.model_validate_json(line))
                except ValidationError as e:
                    errors.append(f"Line {i}: {e}")

        if errors and not result:
            raise SerdeError(  # noqa: TRY301
                f"All lines in {file} failed validation for {get_full_type_name(type_)}. "
                f"First few errors: {'; '.join(errors[:3])}"
            )

    except ValidationError as e:
        raise SerdeError(
            f"Data from {file} is not valid for {get_full_type_name(type_)}"
        ) from e
    except Exception as e:
        line_num = getattr(e, "lineno", "unknown")
        raise SerdeError(f"Invalid JSON at line {line_num} in {file}: {e}") from e

    return result


def load_data[T: BaseModel](file: Path | bytes, type_: type[T]) -> list[T]:
    """Load list of objects from the JSON `file`.

    Args:
        file: File to read the data from, or the actual data in bytes form.
        type_: Type of the objects in the list.
        use_alias: If True, read object keys by using the real field names, not aliases.

    Returns:
        List of data with the `type_` format.

    Raises:
        `ValidationError` if file already exists and its data is incompatible with
        `type_`.
        `OSError` if the file operations fail.
    """
    if isinstance(file, Path):
        content = read_file_bytes(file)
        source = file
    else:
        content = file
        source = "bytes"

    data_raw = orjson.loads(content)
    try:
        return [type_.model_validate(item) for item in data_raw]
    except orjson.JSONDecodeError as e:
        raise SerdeError(f"Data from {source} is not valid JSON.") from e
    except ValidationError as e:
        raise SerdeError(
            f"Data from {source} is not valid for {get_full_type_name(type_)}"
        ) from e


def load_data_single[T: BaseModel](file: Path | bytes, type_: type[T]) -> T:
    """Load a single object from the JSON `file`.

    Args:
        file: File to read the data from, or the actual data in bytes form.
        type_: Type of the object.

    Returns:
        Object with the `type_` format.

    Raises:
        `ValidationError` if file already exists and its data is incompatible with
        `type_`.
        `OSError` if the file operations fail.
    """
    if isinstance(file, Path):
        content = read_file_bytes(file)
    else:
        content = file

    try:
        return type_.model_validate_json(content)
    except ValidationError:
        source = file if isinstance(file, Path) else "bytes"
        raise SerdeError(
            f"Data from {source} is not valid for {get_full_type_name(type_)}"
        ) from None


def save_data[T: BaseModel](
    file: Path,
    data: Sequence[T] | T | Any,
    use_alias: bool = True,
    compress: Compress | None = Compress.ZSTD,
) -> None:
    """Save data in JSON `file`. Can be a single Pydantic object or a Sequence, or Any.

    If `data` is a Pydantic object or Sequence of one, we'll use Pydantic to convert
    to JSON. If not, we'll use `json.dumps` directly.

    Args:
        file: File where data will be saved. Creates its parent directory if it doesn't
            exist. Extension will be added based on compression type.
        data: The data to be saved. If it's a sequence, it must be non-empty.
        use_alias: If True, the output object keys will use the field alias, not the
            actual field name.
        compress: Compression type to use. If None, will infer from the file path.

    Raises:
        SerdeError: if `data` is empty.
    """
    if not data:
        raise SerdeError("Cannot save empty data")

    json_bytes = _dump_data_to_json(data, use_alias=use_alias)
    write_file_bytes(file, json_bytes, compress=compress)


def _dump_data_to_json[T: BaseModel](
    data: Sequence[T] | T | Any, use_alias: bool
) -> bytes:
    """Return a JSON string representation of `data`.

    - If `data` is a single BaseModel, use its `.model_dump_json()`.
    - If `data` is a non-empty sequence of BaseModel, use `TypeAdapter`.
    - Otherwise, fall back to `json.dumps`.
    """

    if isinstance(data, BaseModel):
        return data.model_dump_json(by_alias=use_alias).encode()

    if isinstance(data, Sequence):
        data = cast(Sequence[Any], data)
        if _is_model_list(data):
            type_ = type(data[0])
            return TypeAdapter(Sequence[type_]).dump_json(data)

    return orjson.dumps(data)


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
    modules = sys.modules.copy()
    for module_name, module in modules.items():
        if hasattr(module, type_.__name__):
            obj = getattr(module, type_.__name__)
            if obj is type_:
                return f"{module_name}.{type_.__qualname__}"

    # Fallback to default
    return f"{type_.__module__}.{type_.__qualname__}"


def safe_load_json(file_path: Path) -> Any:
    """Load a JSON file, removing invalid UTF-8 characters."""
    # Decode with error replacement to handle invalid UTF-8
    text = read_file_bytes(file_path).decode("utf-8", errors="replace")
    return orjson.loads(text)


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
