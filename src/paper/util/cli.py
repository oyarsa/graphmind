"""Utilities to handle CLI operations."""

import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any, NoReturn

import click


def die(message: Any, code: int = 1, prefix: str | None = "Error:") -> NoReturn:
    """Print `message` and exit with an error.

    If the SHOW_TRACE environment variable is 1 and there's an exception in flight, print
    the full stack trace.

    Args:
        message: Message to be printed to stderr before quitting.
        code: Error code to exit with.
        prefix: Print before the message. Defaults to `Error: {msg}`.

    Returns:
        NoReturn: quits the program with `code`.
    """
    show_trace = os.environ.get("SHOW_TRACE") == "1"
    is_exc = sys.exc_info()[1] is not None
    if show_trace and is_exc:
        traceback.print_exc()
        print(file=sys.stderr)

    if prefix:
        print(prefix, end=" ")
    print(message, file=sys.stderr)
    sys.exit(code)


def choice(choices: Iterable[str]) -> click.Choice:
    """Create a `click.Choice` by sorting an iterable."""
    return click.Choice(sorted(choices))
