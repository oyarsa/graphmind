"""Load .env file to global environment.

Portions of this file are derived from python-dotenv
Copyright (c) 2014, Saurabh Kumar (python-dotenv)
Copyright (c) 2013, Ted Tieken (django-dotenv-rw)
Copyright (c) 2013, Jacob Kaplan-Moss (django-dotenv)

Original code licensed under BSD-3-Clause License
See: https://github.com/theskumar/python-dotenv/blob/16f2bdad2ebbaae72790514cce713d2d22ab0f7c/LICENSE
"""

from __future__ import annotations

import codecs
import os
import re
from collections import ChainMap
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO


def load_dotenv(
    filename: str = ".env", interpolate: bool = True, override: bool = False
) -> None:
    """Load variables from dotenv files, merging from CWD up to project root.

    For filename=".env", searches for both .env and .env.local files.
    Variables from closer files override those from more distant files.
    .env.local files take precedence over .env files at the same level.

    Variables loaded are applied to os.environ.

    Args:
        filename: Name of the file to search for (default: ".env").
        interpolate: Whether to perform variable interpolation (default: True).
        override: If True, new values override existing os.environ during interpolation
            (default: False).

    Raises:
        FileNotFoundError: if no matching file is found.
    """
    start_path = Path.cwd()
    project_root = _find_project_root(start_path)

    if project_root is not None:
        # Multi-file approach: collect and merge files from CWD to project root
        dotenv_files = _collect_dotenv_files(start_path, project_root, filename)
        if not dotenv_files:
            raise FileNotFoundError(
                f"Could not find any {filename!r} or {filename!r}.local files "
                f"from {str(start_path)!r} up to project root {str(project_root)!r}."
            )
        values = _merge_dotenv_values(
            dotenv_files=dotenv_files,
            encoding="utf-8",
            interpolate=interpolate,
            override=override,
        )
    else:
        # Fallback: single-file approach
        dotenv_path = _find_dotenv(filename)
        values = _dotenv_values(
            dotenv_path=dotenv_path,
            encoding="utf-8",
            interpolate=interpolate,
            override=override,
        )

    # Apply to environment
    for k, v in values.items():
        if not override and k in os.environ:
            continue
        if v is not None:
            os.environ[k] = v


def _find_dotenv(filename: str) -> Path:
    """Search upward from CWD for `filename`.

    Raises:
        FileNotFoundError: if not found.

    Returns:
        Absolute path to the file.
    """
    start = Path.cwd()

    for dirname in _walk_to_root(start):
        check_path = dirname / filename
        if check_path.is_file():
            return check_path

    raise FileNotFoundError(
        f"Could not find {filename!r} starting at {str(start)!r} and walking up the tree."
    )


def _find_project_root(start_path: Path) -> Path | None:
    """Find project root by looking for .git (file or directory) walking upward.

    Args:
        start_path: Starting directory to search from.

    Returns:
        Path to project root directory, or None if not found.
    """
    for dirname in _walk_to_root(start_path):
        if (dirname / ".git").exists():
            return dirname
    return None


def _collect_dotenv_files(
    start_path: Path, project_root: Path, base_filename: str
) -> list[Path]:
    """Collect .env and .env.local files from start_path up to project_root.

    Args:
        start_path: Starting directory to search from.
        project_root: Project root directory (inclusive).
        base_filename: Base filename (e.g., ".env").

    Returns:
        List of existing dotenv files in precedence order (closest first).
    """
    files: list[Path] = []

    for dirname in _walk_to_root(start_path):
        # Check .local first (higher precedence)
        local_path = dirname / f"{base_filename}.local"
        if local_path.is_file():
            files.append(local_path)

        # Check base file
        base_path = dirname / base_filename
        if base_path.is_file():
            files.append(base_path)

        # Stop at project root
        if dirname == project_root:
            break

    return files


def _merge_dotenv_values(
    dotenv_files: list[Path],
    encoding: str,
    interpolate: bool,
    override: bool,
) -> dict[str, str | None]:
    """Read and merge multiple .env files with proper precedence.

    Args:
        dotenv_files: List of dotenv files in precedence order (highest first).
        encoding: File encoding to use.
        interpolate: Whether to perform variable interpolation.
        override: Precedence during interpolation.

    Returns:
        Dict of merged environment variable assignments.
    """
    if not dotenv_files:
        return {}

    # Parse each file into separate dicts (skip files that can't be read)
    file_dicts: list[dict[str, str | None]] = []
    for dotenv_path in dotenv_files:
        try:
            # Read raw values without interpolation
            with dotenv_path.open(encoding=encoding) as stream:
                raw_items = _parse_stream(stream)
                raw_values = {b.key: b.value for b in raw_items if b.key is not None}
                file_dicts.append(raw_values)
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    if not file_dicts:
        return {}

    # Merge using ChainMap (first dict wins)
    merged_values = dict(ChainMap(*file_dicts))

    if not interpolate:
        return merged_values

    return _resolve_variables(merged_values.items(), override=override)


def _walk_to_root(path: Path) -> Iterator[Path]:
    """Yield directories starting from the given directory up to the filesystem root."""
    if not path.exists():
        raise OSError("Starting path not found")

    if path.is_file():
        path = path.parent

    last_dir: Path | None = None
    current_dir = path.resolve()
    while last_dir != current_dir:  # Current dir is not root
        yield current_dir
        last_dir, current_dir = current_dir, current_dir.parent


def _dotenv_values(
    dotenv_path: Path,
    encoding: str | None,
    interpolate: bool,
    override: bool,
) -> dict[str, str | None]:
    """Read a .env file and return a dict of key -> value (possibly interpolated).

    Args:
        dotenv_path: Path to .env file (must exist).
        encoding: File encoding to use.
        interpolate: Whether to perform ${VAR} / ${VAR:-default} interpolation.
        override: Precedence during interpolation:
            - If True:  new values override os.environ when expanding.
            - If False: os.environ has precedence over new values when expanding.

    Returns:
        Dict of parsed environment variable assignments.
    """
    with dotenv_path.open(encoding=encoding) as stream:
        raw_items = _parse_stream(stream)
        raw_values = {b.key: b.value for b in raw_items if b.key is not None}

    if not interpolate:
        return raw_values

    return _resolve_variables(raw_values.items(), override=override)


def _resolve_variables(
    values: Iterable[tuple[str, str | None]],
    override: bool,
) -> dict[str, str | None]:
    """Interpolate ${VAR} and ${VAR:-default} in the provided values."""
    new_values: dict[str, str | None] = {}

    for name, value in values:
        if value is None:
            result = None
        else:
            atoms = _parse_variables(value)
            env: dict[str, str | None] = {}
            if override:
                env.update(os.environ)
                env.update(new_values)
            else:
                env.update(new_values)
                env.update(os.environ)
            result = "".join(atom.resolve(env) for atom in atoms)

        new_values[name] = result

    return new_values


@dataclass(frozen=True)
class _Original:
    string: str
    line: int


@dataclass(frozen=True)
class _Binding:
    key: str | None
    value: str | None
    original: _Original
    error: bool


def _parse_stream(stream: IO[str]) -> Iterator[_Binding]:
    reader = _Reader(stream)
    while reader.has_next():
        yield _parse_binding(reader)


class _Reader:
    def __init__(self, stream: IO[str]) -> None:
        self.string = stream.read()
        self.position = _Position.start()
        self.mark = _Position.start()

    def has_next(self) -> bool:
        return self.position.chars < len(self.string)

    def set_mark(self) -> None:
        self.mark.set(self.position)

    def get_marked(self) -> _Original:
        return _Original(
            string=self.string[self.mark.chars : self.position.chars],
            line=self.mark.line,
        )

    def peek(self, count: int) -> str:
        return self.string[self.position.chars : self.position.chars + count]

    def read_regex(self, regex: re.Pattern[str]) -> tuple[str, ...]:
        match = regex.match(self.string, self.position.chars)
        if match is None:
            raise _ParseError("read_regex: Pattern not found")
        self.position.advance(self.string[match.start() : match.end()])
        return match.groups()


class _Position:
    def __init__(self, chars: int, line: int) -> None:
        self.chars = chars
        self.line = line

    @classmethod
    def start(cls) -> _Position:
        return cls(chars=0, line=1)

    def set(self, other: _Position) -> None:
        self.chars = other.chars
        self.line = other.line

    def advance(self, string: str) -> None:
        self.chars += len(string)
        self.line += len(re.findall(_newline, string))


def _make_regex(string: str, extra_flags: int = 0) -> re.Pattern[str]:
    return re.compile(string, re.UNICODE | extra_flags)


_newline = _make_regex(r"(\r\n|\n|\r)")
_multiline_whitespace = _make_regex(r"\s*", extra_flags=re.MULTILINE)
_whitespace = _make_regex(r"[^\S\r\n]*")
_export = _make_regex(r"(?:export[^\S\r\n]+)?")
_single_quoted_key = _make_regex(r"'([^']+)'")
_unquoted_key = _make_regex(r"([^=\#\s]+)")
_equal_sign = _make_regex(r"(=[^\S\r\n]*)")
_single_quoted_value = _make_regex(r"'((?:\\'|[^'])*)'")
_double_quoted_value = _make_regex(r'"((?:\\"|[^"])*)"')
_unquoted_value = _make_regex(r"([^\r\n]*)")
_comment = _make_regex(r"(?:[^\S\r\n]*#[^\r\n]*)?")
_end_of_line = _make_regex(r"[^\S\r\n]*(?:\r\n|\n|\r|$)")
_rest_of_line = _make_regex(r"[^\r\n]*(?:\r|\n|\r\n)?")
_double_quote_escapes = _make_regex(r"\\[\\'\"abfnrtv]")
_single_quote_escapes = _make_regex(r"\\[\\']")
_posix_variable = re.compile(
    r"""
    \$\{
        (?P<name>[^\}:]*)
        (?::-
            (?P<default>[^\}]*)
        )?
    \}
    """,
    re.VERBOSE,
)


class _ParseError(Exception):
    """Error raised when loading or parsing .env files."""


def _parse_variables(value: str) -> Iterator[_Literal | _Variable]:
    cursor = 0

    for match in _posix_variable.finditer(value):
        start, end = match.span()
        name = match["name"]
        default = match["default"]

        if start > cursor:
            yield _Literal(value=value[cursor:start])

        yield _Variable(name=name, default=default)
        cursor = end

    if cursor < len(value):
        yield _Literal(value=value[cursor:])


class _Literal:
    def __init__(self, value: str) -> None:
        self.value = value

    def resolve(self, env: Mapping[str, str | None]) -> str:
        return self.value


class _Variable:
    def __init__(self, name: str, default: str | None) -> None:
        self.name = name
        self.default = default

    def resolve(self, env: Mapping[str, str | None]) -> str:
        default = self.default if self.default is not None else ""
        result = env.get(self.name, default)
        return result if result is not None else ""


def _parse_key(reader: _Reader) -> str | None:
    match reader.peek(1):
        case "#":
            return None
        case "'":
            return reader.read_regex(_single_quoted_key)[0]
        case _:
            return reader.read_regex(_unquoted_key)[0]


def _decode_escapes(regex: re.Pattern[str], string: str) -> str:
    def decode_match(match: re.Match[str]) -> str:
        return codecs.decode(match.group(0), "unicode-escape")

    return regex.sub(decode_match, string)


def _parse_value(reader: _Reader) -> str:
    match reader.peek(1):
        case "'":
            (value,) = reader.read_regex(_single_quoted_value)
            return _decode_escapes(_single_quote_escapes, value)
        case '"':
            (value,) = reader.read_regex(_double_quoted_value)
            return _decode_escapes(_double_quote_escapes, value)
        case "" | "\n" | "\r":
            return ""
        case _:
            return _parse_unquoted_value(reader)


def _parse_unquoted_value(reader: _Reader) -> str:
    (part,) = reader.read_regex(_unquoted_value)
    return re.sub(r"\s+#.*", "", part).rstrip()


def _parse_binding(reader: _Reader) -> _Binding:
    reader.set_mark()

    try:
        reader.read_regex(_multiline_whitespace)

        if not reader.has_next():
            return _Binding(
                key=None,
                value=None,
                original=reader.get_marked(),
                error=False,
            )

        reader.read_regex(_export)
        key = _parse_key(reader)
        reader.read_regex(_whitespace)

        if reader.peek(1) == "=":
            reader.read_regex(_equal_sign)
            value = _parse_value(reader)
        else:
            value = None

        reader.read_regex(_comment)
        reader.read_regex(_end_of_line)

        return _Binding(
            key=key,
            value=value,
            original=reader.get_marked(),
            error=False,
        )
    except _ParseError:
        reader.read_regex(_rest_of_line)

        return _Binding(
            key=None,
            value=None,
            original=reader.get_marked(),
            error=True,
        )
