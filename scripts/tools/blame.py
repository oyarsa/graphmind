#!/usr/bin/env python3
"""Nice formatting of git blame with short hash, date, commit summary and line.

Usage:
    $ python blame.py ARGS

Where ARGS are standard git blame arguments (e.g. --follow and the file path).
They are forwarded straight to git blame.
"""

import datetime as dt
import subprocess
import sys
from dataclasses import dataclass

import click

COLOURS = {
    "hash": "\033[36m",  # cyan
    "date": "\033[32m",  # green
    "summary": "\033[33m",  # yellow
    "line_num": "\033[35m",  # magenta
    "reset": "\033[0m",  # reset to default
}


@dataclass
class CurrentCommit:
    """Commit being parsed from porcelain output."""

    hash: str | None = None
    date: str | None = None
    summary: str | None = None
    line_num: int | None = None

    def is_complete(self) -> bool:
        """Check if all fields have been filled."""
        return all(
            getattr(self, field) is not None
            for field in ["hash", "date", "summary", "line_num"]
        )


@dataclass(frozen=True, kw_only=True)
class FormattedLine:
    """Information to build the formatted line."""

    hash: str
    date: str
    summary: str
    line_num: int
    content: str


def _get_column_widths(formatted_lines: list[FormattedLine]) -> dict[str, int]:
    """Calculate the maximum width for each column."""
    return {
        "hash": 7,  # hash is always 7 chars
        "date": 10,  # date is always YYYY-MM-DD
        "summary": max(len(line.summary.strip()) for line in formatted_lines),
        "line_num": len(str(max(line.line_num for line in formatted_lines))),
    }


def main() -> None:
    """Format git blame output nicely, including other CLI arguments."""
    cmd = ["git", "blame", "--porcelain"] + sys.argv[1:]

    # Run git blame and capture output
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)

    current_commit = CurrentCommit()
    formatted_lines: list[FormattedLine] = []

    for line in result.stdout.splitlines():
        if not line:
            continue

        if line.startswith("\t"):
            # This is the actual file line
            content = line[1:]  # Remove the tab
            if current_commit.is_complete():  # Only process if we have commit info
                formatted_lines.append(
                    FormattedLine(
                        hash=current_commit.hash[:7],  # type: ignore
                        date=current_commit.date,  # type: ignore
                        summary=current_commit.summary,  # type: ignore
                        line_num=current_commit.line_num,  # type: ignore
                        content=content,
                    )
                )
        # Parse metadata lines
        elif line.startswith("author-time "):
            timestamp = int(line.split()[1])
            current_commit.date = dt.datetime.fromtimestamp(
                timestamp, tz=dt.UTC
            ).strftime("%Y-%m-%d")
        elif line.startswith("summary "):
            current_commit.summary = line[8:]  # Remove 'summary '
        else:
            # First line contains hash and line number in first two positions
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 40:  # Full SHA-1 hash is 40 chars
                try:
                    current_commit.hash = parts[0]
                    current_commit.line_num = int(parts[1])
                except ValueError:
                    # Skip lines that don't match our expected format
                    continue

    widths = _get_column_widths(formatted_lines)
    output: list[str] = []
    for line in formatted_lines:
        output.append(
            f"{_style(line.hash, "hash", widths)} "
            f"{_style(line.date, "date", widths)} "
            f"{_style(line.summary, "summary", widths)} "
            f"{_style(line.line_num, "line_num", widths, '>')} "
            f"{line.content}\n"
        )
    click.echo_via_pager(output)


def _style(value: object, field: str, widths: dict[str, int], align: str = "<") -> str:
    """Format field as string with the specified colour and determined width."""
    return f"{COLOURS[field]}{value:{align}{widths[field]}}{COLOURS['reset']}"


if __name__ == "__main__":
    main()
