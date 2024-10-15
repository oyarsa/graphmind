"""Infamous utility module for stuff I can't place anywhere else."""

import time
from typing import Any, Self

from thefuzz import fuzz  # type: ignore


def fuzzy_ratio(s1: str, s2: str) -> int:
    """Calculates the fuzzy matching ratio between s1 and s2 as integer in 0-100.

    Type-safe wrapper around thefuzz.fuzz.ratio.
    """
    return fuzz.ratio(s1, s2)  # type: ignore


def convert_time_elapsed(seconds: float) -> str:
    """Convert a time duration from seconds to a human-readable format."""
    units = [("d", 86400), ("h", 3600), ("m", 60)]
    parts: list[str] = []

    for name, count in units:
        value, seconds = divmod(seconds, count)
        if value >= 1:
            parts.append(f"{int(value)}{name}")

    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f}s")

    return " ".join(parts)


class BlockTimer:
    """Track the time elapsed during its block."""

    def __init__(self) -> None:
        self._start_time = 0
        self._elapsed_seconds = 0

    def __enter__(self) -> Self:
        """Start the timer when entering the context."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Stop the timer when exiting the context."""
        self._elapsed_seconds = time.perf_counter() - self._start_time

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
