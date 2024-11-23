"""Infamous utility module for stuff I can't place anywhere else."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import sys
import time
from collections.abc import Callable, Coroutine
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import colorlog
from thefuzz import fuzz  # type: ignore

from paper.util import cli


def fuzzy_ratio(s1: str, s2: str) -> int:
    """Calculate the fuzzy matching ratio between s1 and s2 as integer in 0-100.

    Type-safe wrapper around thefuzz.fuzz.ratio.
    """
    return fuzz.ratio(s1, s2)  # type: ignore


def fuzzy_partial_ratio(s1: str, s2: str) -> int:
    """Calculate the partial fuzzy matching ratio between s1 and s2 as integer in 0-100.

    Type-safe wrapper around thefuzz.fuzz.partial_ratio.
    """
    return fuzz.partial_ratio(s1, s2)  # type: ignore


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
        traceback: TracebackType | None,
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


def setup_logging() -> None:
    """Initialise `paper` and `__main__` loggers, printing colourful output to stderr.

    Uses `LOG_LEVEL` environment variable to set the level. By default, it's INFO. Use
    the standard level names (see documentation for `logging` module).

    This function initialises both the package's root logger `paper` and the `__main__`
    logger.
    """
    _setup_logging("paper")  # Set up loggers for imported packages
    _setup_logging("__main__")  # This one's for when a script is called directly


def _setup_logging(logger_name: str) -> None:
    """Initialise a logger printing colourful output to stderr.

    Uses `LOG_LEVEL` environment variable to set the level. By default, it's INFO. Use
    the standard level names (see documentation for `logging` module).

    Args:
        logger_name: String used to locate a global logger. Initialises the global
            project logger, including all its descendants.
    """
    logger = logging.getLogger(logger_name)

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
    except FileNotFoundError:
        return "new"
    except Exception:
        return "error"


def mustenv(*variables: str) -> dict[str, str]:
    """Ensure that all environment `variables` exist and return a dict with the values.

    If any variables are unset, print an error with them and exit with code 1.
    """
    vars_ = {var: os.environ.get(var) for var in variables}

    if vars_unset := sorted(var for var, value in vars_.items() if not value):
        cli.die(
            "The following required environment variables were unset:"
            f" {", ".join(vars_unset)}."
        )

    return {var: value for var, value in vars_.items() if value}


def ensure_envvar(name: str) -> str:
    """Get an environment variable or print a nice error if unavailable.

    Args:
        name: name of the environment variable, e.g. `OPENAI_API_KEY`.

    Returns:
        The value of the variable if it's set. A variable set to the empty string counts
        as unset.

    Raises:
        SystemExit: if the variable isn't set, prints a nice error and quits.
    """
    return mustenv(name)[name]


def run_safe[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run `func` until it ends, or the user quits with Ctrl-C, with confirmation.

    This means a single Ctrl-C won't quit; the user will be prompted to ensure they
    really want to quit.

    Args:
        func: Function that's called with `args` and `kwargs`. The return value is
            returned if the user doesn't quit.
        args: Positional arguments for `func`.
        kwargs: Keyword arguments for `func`.
    """
    while True:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            choice = input("\n\nCtrl+C detected. Do you really want to exit? (y/n): ")
            if choice.lower() == "y":
                sys.exit()
            else:
                # The loop will continue, restarting _download
                print("Continuing...\n")


def arun_safe[**P, R](
    async_func: Callable[P, Coroutine[Any, Any, R]], *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run `async_func` until it ends, or the user quits with Ctrl-C, with confirmation.

    This means a single Ctrl-C won't quit; the user will be prompted to ensure they
    really want to quit. The function is executed using `asyncio.run` with the default
    parameters.

    Args:
        async_func: Function that's called with `args` and `kwargs`. The return value is
            returned if the user doesn't quit.
        args: Positional arguments for `func`.
        kwargs: Keyword arguments for `func`.
    """
    return run_safe(asyncio.run, async_func(*args, **kwargs))
