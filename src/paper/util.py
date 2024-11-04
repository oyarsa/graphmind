"""Infamous utility module for stuff I can't place anywhere else."""

from __future__ import annotations

import copy
import hashlib
import heapq
import inspect
import logging
import os
import time
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, Self

import colorlog
from thefuzz import fuzz  # type: ignore


def fuzzy_ratio(s1: str, s2: str) -> int:
    """Calculates the fuzzy matching ratio between s1 and s2 as integer in 0-100.

    Type-safe wrapper around thefuzz.fuzz.ratio.
    """
    return fuzz.ratio(s1, s2)  # type: ignore


class Timer:
    """Track time elapsed. Can be used as a context manager to time its block."""

    def __init__(self) -> None:
        self._start_time = 0
        self._elapsed_seconds = 0

    def start(self) -> None:
        """(Re)start the timer."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer."""
        self._elapsed_seconds = time.perf_counter() - self._start_time

    def __enter__(self) -> Self:
        """Start the timer when entering the context."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Stop the timer when exiting the context."""
        self.stop()

    @property
    def seconds(self) -> float:
        """Return the elapsed time in seconds."""
        return self._elapsed_seconds

    @property
    def human(self) -> str:
        """Return the elapsed time in a human-readable format."""
        seconds = self._elapsed_seconds
        units = [("d", 86400), ("h", 3600), ("m", 60)]
        parts: list[str] = []

        for name, count in units:
            value, seconds = divmod(seconds, count)
            if value >= 1:
                parts.append(f"{int(value)}{name}")

        if seconds > 0 or not parts:
            parts.append(f"{seconds:.2f}s")

        return " ".join(parts)


def setup_logging(logger: logging.Logger | str = "paper") -> None:
    """Initialise a logger printing colourful output to stderr.

    Uses `LOG_LEVEL` environment variable to set the level. By default, it's INFO. Use
    the standard level names (see documentation for `logging` module).

    Args:
        logger: A proper Logger object, or a string to locate one. By default,
            initialises the global project logger, including all its descendants.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = colorlog.StreamHandler()

    fmt = "%(log_color)s%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(colorlog.ColoredFormatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(handler)


def read_resource(package: str, filename: str) -> str:
    """Read text from resource file.

    Args:
        package: Path to package, relative to `paper`.
        filename: Name of the file under `paper.{package}`.
    """
    return resources.files(f"paper.{package}").joinpath(filename).read_text()


def safediv(x: float, y: float) -> float:
    """x/y where if `y` is 0, returns NaN instead of throwing.

    Args:
        x: numerator
        y: denominator (can be 0)

    Returns:
        If y is not 0, returns x/y. Else, returns NaN.
    """
    try:
        return x / y
    except ZeroDivisionError:
        return float("nan")


class Comparable(Protocol):
    """Protocol for comparable types. Requires `<` (__lt__)."""

    def __lt__(self: Self, other: Self, /) -> bool: ...


class TopKSet[T: Comparable]:
    """Set that keeps the top K items. The item type needs to be `Comparable`.

    If the collection has less than K items, it accepts any new item. Once K is reached,
    only items that are larger than the smallest of the current items on the list are
    added.

    Items added are also added to a set to make them unique. Note that this means that
    `T` must be hashable.

    Access the items on the list via the `items` property, which returns a new list.
    """

    def __init__(self, type_: type[T], k: int) -> None:
        """Initialise TopK list.

        Args:
            type_: Type of the elements of the list.
            k: How many items to keep.
        """
        self.k = k
        self.data: list[T] = []
        self.seen: set[T] = set()

    def add(self, item: T) -> None:
        """Add new item to collection, depending on the value of `item`.

        - If we have less than k items, just add it.
        - If the new item is larger than the smallest in the list, replace it.
        - Otherwise, ignore it.
        """
        if item in self.seen:
            return
        self.seen.add(item)

        if len(self.data) < self.k:
            heapq.heappush(self.data, item)
        elif item > self.data[0]:
            heapq.heapreplace(self.data, item)

    @property
    def items(self) -> list[T]:
        """Items from the collection in a new list, sorted by descending value.

        Both the list and the items are new, so no modifications will affect the
        collection.

        NB: The list is new by construction, and the items are copied with `deepcopy`.
        """
        return [copy.deepcopy(item) for item in sorted(self.data, reverse=True)]


def display_params() -> str:
    """Display the function parameters and values as a formatted string.

    Masks sensitive values, i.e. values with `api` in the name.

    Returns:
        String representation of the parameter names and values.

    Raises:
        ValueError: if there's a problem getting the calling function object.
    """
    if (curframe := inspect.currentframe()) and curframe.f_back:
        frame = curframe.f_back
    else:
        raise ValueError("Couldn't find calling function frame")

    args = frame.f_locals
    func_name = frame.f_code.co_name

    # Get the actual function object from the frame
    for obj in frame.f_globals.values():
        if inspect.isfunction(obj) and obj.__name__ == func_name:
            func = obj
            break
    else:
        raise ValueError(f"Couldn't find function object '{func_name}'")

    result: dict[str, str] = {}

    for param_name, param in inspect.signature(func).parameters.items():
        value = args.get(param_name, param.default)

        if isinstance(value, Path):
            value = f"{value.resolve()} ({_hash_path(value)})"

        # Mask sensitive values
        if "api" in param_name.casefold() and value is not None:
            result[param_name] = "********"
        else:
            result[param_name] = str(value)

    return (
        "CONFIG:\n"
        + "\n".join(f"{key}: {value}" for key, value in result.items())
        + "\n"
    )


def _hash_path(path: Path, chars: int = 8) -> str:
    """Calculate truncated SHA-256 hash of a path if it's a file.

    If it's a directory, returns `directory`. Otherwise, returns `error`.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:chars]
    except IsADirectoryError:
        return "directory"
    except Exception:
        return "error"
