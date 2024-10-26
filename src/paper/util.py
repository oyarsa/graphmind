"""Infamous utility module for stuff I can't place anywhere else."""

import logging
import os
import time
import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Any, Self

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


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    type_name: str
    template: str


def load_prompts(name: str) -> dict[str, PromptTemplate]:
    """Load prompts from a TOML file in the prompts package.

    Args:
        name: Name of the TOML file in `paper.gpt.prompts`, without extension.

    Returns:
        Dictionary mapping prompt names to their text content.
    """
    text = read_resource("gpt.prompts", f"{name}.toml")
    return {
        p["name"]: PromptTemplate(p["name"], p["type"], p["prompt"])
        for p in tomllib.loads(text)["prompts"]
    }


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
