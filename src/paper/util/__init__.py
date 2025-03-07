"""Infamous utility module for stuff I can't place anywhere else."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, Self, overload

import psutil
from thefuzz import fuzz  # type: ignore

from paper.util import cli

logger = logging.getLogger(__name__)


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

    def __init__(self, name: str | None = None) -> None:
        self._start_time = 0
        self._elapsed_seconds = 0
        self.name = name

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

    def __str__(self) -> str:
        """Return "{name} took {time}"."""
        name = self.name if self.name else "Timer"
        return f"{name} took {self.human}"


def setup_logging() -> None:
    """Initialise `paper` and `__main__` loggers, printing colourful output to stderr.

    Uses `LOG_LEVEL` environment variable to set the minimum level. By default, it's INFO.
    Use the standard level names (see documentation for `logging` module). The output to
    the terminal has different colours per log level.

    This function initialises both the package's root logger `paper` and the `__main__`
    logger, in case you're running a script directly.
    """
    _setup_logging("paper")  # Set up loggers for imported packages
    _setup_logging("__main__")  # This one's for when a script is called directly


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter to add colors to log levels."""

    COLORS: Mapping[str, str] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[0m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    """ANSI colour codes for each log level."""
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the logging output with colours according to the message log level."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.log_color = log_color
        record.reset = self.RESET
        formatted_message = super().format(record)
        return f"{log_color}{formatted_message}{self.RESET}"


def _setup_logging(logger_name: str) -> None:
    """Initialise a logger printing colourful output to stderr.

    Uses `LOG_LEVEL` environment variable to set the level. By default, it's INFO. Use
    the standard level names (see documentation for `logging` module).

    Args:
        logger_name: String used to locate a global logger. Initialises the global
            project logger, including all its descendants.
    """
    logger = logging.getLogger(logger_name)

    # Set the log level from the environment variable or default to INFO
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    # Create a stream handler
    handler = logging.StreamHandler(sys.stderr)

    # Define the log format and date format
    fmt = "%(log_color)s%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s%(reset)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Set the custom formatter
    handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
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


def get_params(frame_depth: int = 1) -> dict[str, str]:
    """Get calling function parameters and values as a dictionary.

    Masks sensitive values, i.e. values with `api` in the name.

    Args:
        frame_depth: Number of frames to go up to find the target function. 1 gets
            direct caller, 2 gets caller's caller, etc.

    Returns:
        Mapping parameter names to their string representations.

    Raises:
        ValueError: if there's a problem getting the calling function object.
    """
    if (curframe := inspect.currentframe()) and (frame := curframe.f_back):
        for _ in range(frame_depth - 1):
            if not frame.f_back:
                raise ValueError("Frame depth exceeds call stack")
            frame = frame.f_back
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

        if "api" in param_name.casefold() and value is not None:
            result[param_name] = "********"
        else:
            result[param_name] = str(value)

    return result


def render_params(params: dict[str, str]) -> str:
    """Render function parameters as a string.

    You can use it to render the output of `get_params`.
    """
    return (
        "CONFIG:\n"
        f"Git commit: {git_commit()}\n"
        + "\n".join(f"{key}: {value}" for key, value in params.items())
        + "\n"
    )


def display_params() -> str:
    """Display the calling function parameters and values as a formatted string.

    Masks sensitive values, i.e. values with `api` in the name.

    Returns:
        String representation of the parameter names and values.
    """
    # Get caller of display_params
    return render_params(get_params(frame_depth=2))


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
            f" {', '.join(vars_unset)}."
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


def git_commit() -> str:
    """Return the current git commit hash, or '<no repo>' if not in a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd(),
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "<no repo>"


def git_root() -> Path:
    """Return the absolute path to the root of the current git repository.

    Raises:
        subprocess.SubprocessError: If the current directory is not part of a git
            repository.
    """
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(git_root).absolute()
    except subprocess.CalledProcessError as e:
        raise subprocess.SubprocessError(
            "Current directory is not part of a git repository"
        ) from e


def hashstr(s: str) -> str:
    """Hash string using sha256."""
    return hashlib.sha256(s.encode()).hexdigest()


def shuffled[T](iterable: Iterable[T]) -> list[T]:
    """Return a shallow copy of the contents in `iterable` shuffled as a list.

    This uses standard library's `random`. You can use `random.seed` to initialise the
    seed for this.
    """
    lst = list(iterable)
    random.shuffle(lst)
    return lst


def groupby[T, K](
    iterable: Iterable[T], key: Callable[[T], K | None]
) -> dict[K, list[T]]:
    """Group items into a dict by key function. Ignores items with None key.

    Args:
        iterable: Iterable of items to group.
        key: Function that takes an element and returns the key to group by. If the
            returned key is None, the item is discarded.

    Returns:
        Dictionary where keys are the result of applying the key function to the items,
        and values are lists of items that share the same. Keeps the original order of
        items in each group. Keys are kept in the same order as they are first seen.
    """
    groups: dict[K, list[T]] = {}

    for item in iterable:
        k = key(item)
        if k is None:
            continue
        if k not in groups:
            groups[k] = []
        groups[k].append(item)

    return groups


@overload
def get_icase[T](data: Mapping[str, T], key: str, default: T) -> T: ...


@overload
def get_icase[T](data: Mapping[str, T], key: str, default: None = None) -> T | None: ...


def get_icase[T](data: Mapping[str, T], key: str, default: T | None = None) -> T | None:
    """Get value from dict using case-insensitive key matching."""
    key_lower = key.lower()
    for k in data:
        if k.lower() == key_lower:
            return data[k]
    return default


def format_numbered_list(
    items: Iterable[str],
    prefix: str = "",
    suffix: str = ".",
    indent: int = 0,
    start: int = 1,
    sep: str = "\n",
) -> str:
    """Format an iterable of strings as a numbered list.

    Args:
        items: An iterable containing strings to be formatted.
        prefix: Text to add before the number. Useful if you have a nested list, e.g.
            `1.1.`
        suffix: The bullet symbol to use after the number.
        indent: Number of spaces to indent the entire list.
        start: Starting index for number label.
        sep: Separator to join lines.

    Returns:
        A formatted string where each item appears on a new line, properly indented and
        prefixed with the bullet symbol and the number.
    """
    base_indent = " " * indent
    return sep.join(
        f"{base_indent}{prefix}{i}{suffix} {item}"
        for i, item in enumerate(items, start=start)
    )


def format_bullet_list(items: Iterable[str], prefix: str = "-", indent: int = 0) -> str:
    """Format an iterable of strings as a bullet list.

    Args:
        items: An iterable containing strings to be formatted.
        prefix: The bullet symbol to use (defaults to "-").
        indent: Number of spaces to indent the entire list.

    Returns:
        A formatted string where each item appears on a new line, properly indented and
        prefixed with the bullet symbol
    """
    base_indent = " " * indent
    return "\n".join(f"{base_indent}{prefix} {item}" for item in items)


def remove_parenthetical(text: str) -> str:
    """Remove text within parentheses, including nested ones.

    Also removes consecutive spaces after parentheses are removed.

    Args:
        text: The input string to process.

    Returns:
        The string with parenthetical content removed.

    Examples:
        >>> remove_parenthetical("Example.")
        "Example."
        >>> remove_parenthetical("Another example (with items)")
        "Another example"
    """
    # Stores the indices of the parentheses encountered
    stack: list[int] = []
    # Stores all characters we encounter. When a parenthesis is found, we remove
    # the respective items until the indice of the last parenthesis.
    result: list[str] = []

    for char in text:
        if char == "(":
            stack.append(len(result))
            result.append(char)
        elif char == ")" and stack:
            last_paren = stack.pop()
            result = result[:last_paren]
        else:
            result.append(char)

    text = "".join(result).strip()
    text = re.sub(r"\s+", " ", text)
    return fix_spaces_before_punctuation(text)


def fix_spaces_before_punctuation(text: str) -> str:
    """Remove space before certain punctuation markers."""
    punctuation = [".", "!", "?", ";", ":", ",", ")", "]", "}"]
    for p in punctuation:
        text = re.sub(rf"\s+{re.escape(p)}", p, text)
    return text


def on_exception[T, **P](
    default: T,
    logger: logging.Logger | None = None,
    level: str = "warning",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a function that throws an exception. If it's raised, return the value.

    The exception itself is swallowed. If `logger` is given, it will be logged using
    the specified level (defaults to "warning").
    """

    log = None
    if logger is not None:
        log = getattr(logger, level, None)
        if log is None:
            log = logger.warning
            log("Invalid log level for `on_exception`: %s. Using 'warning'.", level)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception:
                if log is not None:
                    log("Error suppressed with `on_exception`.")
                return default

        return wrapper

    return decorator


def at[T](seq: Sequence[T], idx: int, desc: str, title: str) -> T | None:
    """Get `seq[idx]` if possible, otherwise return None and log warning with `desc`."""
    try:
        return seq[idx]
    except IndexError:
        logger.warning(
            "Invalid index at '%s' (%s): %d out of %d", title, desc, idx, len(seq)
        )
        return None


def print_memory_usage(file: Path) -> None:
    """Print detailed memory usage information.

    1. Current Python process memory usage
    2. Overall system memory usage
    3. Available memory remaining

    Works on both macOS and Linux systems.
    """

    def log(x: str) -> None:
        print(x)
        with file.open("a") as f:
            f.write(x + "\n" + "-" * 80 + "\n")

    process = psutil.Process(os.getpid())

    python_memory_usage = process.memory_info().rss / (1024 * 1024)
    system_memory = psutil.virtual_memory()
    total_memory_gb = system_memory.total / (1024 * 1024 * 1024)
    used_memory_gb = system_memory.used / (1024 * 1024 * 1024)
    available_memory_gb = system_memory.available / (1024 * 1024 * 1024)
    memory_percent = system_memory.percent

    system_name = platform.system()

    log(f"System: {system_name}")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python Process Memory Usage: {python_memory_usage:.2f} MB")
    log("\nSystem Memory Statistics:")
    log(f"  Total Memory: {total_memory_gb:.2f} GB")
    log(f"  Used Memory: {used_memory_gb:.2f} GB ({memory_percent}%)")
    log(f"  Available Memory: {available_memory_gb:.2f} GB")

    if system_name == "Darwin":  # macOS
        swap = psutil.swap_memory()
        swap_total_gb = swap.total / (1024 * 1024 * 1024)
        swap_used_gb = swap.used / (1024 * 1024 * 1024)
        log("\nmacOS Swap Information:")
        log(f"  Total Swap: {swap_total_gb:.2f} GB")
        log(f"  Used Swap: {swap_used_gb:.2f} GB ({swap.percent}%)")

    elif system_name == "Linux":
        log("\nLinux-specific Memory Information:")
        log(f"  Buffers: {system_memory.buffers / (1024 * 1024 * 1024):.2f} GB")  # type: ignore
        log(f"  Cached: {system_memory.cached / (1024 * 1024 * 1024):.2f} GB")  # type: ignore

        swap = psutil.swap_memory()
        swap_total_gb = swap.total / (1024 * 1024 * 1024)
        swap_used_gb = swap.used / (1024 * 1024 * 1024)
        log("\nLinux Swap Information:")
        log(f"  Total Swap: {swap_total_gb:.2f} GB")
        log(f"  Used Swap: {swap_used_gb:.2f} GB ({swap.percent}%)")
