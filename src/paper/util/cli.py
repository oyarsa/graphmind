"""Utilities to handle CLI operations."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any, NoReturn, override

import click
from click import shell_completion

from paper.util.typing import SupportsLT


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
        print(prefix, end=" ", file=sys.stderr)
    print(message, file=sys.stderr)
    sys.exit(code)


class Choice[T: SupportsLT](click.ParamType):
    """Allow only a fixed set of supported values.

    These values are converted to and from strings.
    """

    name = "choice"

    def __init__(self, choices: Iterable[T]) -> None:
        self.choices = sorted(choices)
        self.choices_str = [str(c) for c in self.choices]

    @override
    def to_info_dict(self) -> dict[str, Any]:
        return super().to_info_dict() | {"choices": self.choices}

    @override
    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str:
        choices_str = "|".join(self.choices_str)

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"

    @override
    def get_missing_message(
        self, param: click.Parameter, ctx: click.Context | None
    ) -> str:
        return "Choose from:\n\t{choices}".format(
            choices=",\n\t".join(self.choices_str)
        )

    @override
    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> Any:
        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value
        normed_choices = {choice: choice for choice in self.choices_str}

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = {
                ctx.token_normalize_func(normed_choice): original
                for normed_choice, original in normed_choices.items()
            }

        if normed_value in normed_choices:
            return normed_choices[normed_value]

        choices_repr = ", ".join(map(repr, self.choices_str))
        return self.fail(f"{value!r} is not one of {choices_repr}.", param, ctx)

    @override
    def __repr__(self) -> str:
        return f"Choice({self.choices_str})"

    @override
    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[shell_completion.CompletionItem]:
        return [
            shell_completion.CompletionItem(c)
            for c in self.choices_str
            if c.lower().startswith(incomplete.lower())
        ]
