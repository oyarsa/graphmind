"""Nice formatting of git blame with short hash, date, commit summary and line.

Usage:
    $ python blame.py ARGS

Where ARGS are standard git blame arguments (e.g. --follow and the file path).
They are forwarded straight to git blame.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import click


def main() -> None:
    """Format git blame output nicely, including other CLI arguments."""
    blame_output = _get_blame_output(sys.argv[1:])
    parsed_lines = _parse_blame_lines(blame_output)
    click.echo_via_pager(_format_output(parsed_lines))


def _get_blame_output(args: list[str]) -> str:
    """Run git blame and return its output."""
    cmd = ["git", "blame", "--porcelain", *args]
    return subprocess.run(cmd, capture_output=True, check=True, text=True).stdout


def _parse_blame_lines(blame_output: str) -> list[ParsedLine]:
    """Parse git blame porcelain output into formatted lines."""
    lines = blame_output.splitlines()
    current_commit = CurrentCommit()
    formatted_lines: list[ParsedLine] = []
    commits_cache: dict[str, CompleteCommit] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        if line.startswith("\t"):
            # This is the actual file line
            content = line[1:]  # Remove the tab
            if complete := current_commit.to_complete():
                formatted_lines.append(
                    ParsedLine(
                        hash=complete.hash[:7],
                        date=complete.date,
                        summary=complete.summary,
                        line_num=complete.line_num,
                        content=content,
                    )
                )
        else:
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 40:  # Full SHA-1 hash is 40 chars
                # Check if we've seen this commit before
                if cached := commits_cache.get(parts[0]):
                    current_commit = CurrentCommit(
                        hash=cached.hash,
                        date=cached.date,
                        summary=cached.summary,
                        line_num=int(parts[1]),
                    )
                else:
                    # New commit, need to parse its metadata
                    current_commit, i = _parse_metadata(lines, i)
                    # Cache this commit's metadata if complete
                    if complete := current_commit.to_complete():
                        commits_cache[complete.hash] = complete
        i += 1

    return formatted_lines


def _parse_metadata(lines: list[str], start_index: int) -> tuple[CurrentCommit, int]:
    """Parse commit metadata from porcelain output starting at the given index.

    Returns the parsed commit and the new index to continue parsing from.
    """
    parts = lines[start_index].split()
    current_commit = CurrentCommit(hash=parts[0], line_num=int(parts[1]))

    i = start_index + 1
    while i < len(lines) and not (
        lines[i].startswith("\t")
        or (len(lines[i].split()) >= 2 and len(lines[i].split()[0]) == 40)
    ):
        metadata = lines[i]
        if metadata.startswith("author-time "):
            timestamp = int(metadata.split()[1])
            current_commit.date = dt.datetime.fromtimestamp(
                timestamp, tz=dt.UTC
            ).strftime("%Y-%m-%d")
        elif metadata.startswith("summary "):
            current_commit.summary = metadata[8:]
        i += 1

    return current_commit, i - 1


@dataclass
class CurrentCommit:
    """Commit currently being parsed from porcelain output.

    The object is mutated as new information is obtained from porcelain. Transforms into
    a `CompleteCommit` once it's done.
    """

    hash: str | None = None
    date: str | None = None
    summary: str | None = None
    line_num: int | None = None

    def to_complete(self) -> CompleteCommit | None:
        """Convert to a `CompleteCommit` if all fields are populated, else returns None."""
        if (
            self.hash is None
            or self.date is None
            or self.summary is None
            or self.line_num is None
        ):
            return None

        return CompleteCommit(
            hash=self.hash, date=self.date, summary=self.summary, line_num=self.line_num
        )


@dataclass
class CompleteCommit:
    """Commit with complete metadata."""

    hash: str
    date: str
    summary: str
    line_num: int


@dataclass(frozen=True, kw_only=True)
class ParsedLine:
    """Line information parsed from git."""

    hash: str
    date: str
    summary: str
    line_num: int
    content: str


def _format_output(lines: list[ParsedLine]) -> Iterator[str]:
    """Format parsed lines for display.

    Yields:
        Lines for the output one at a time.
    """
    widths = _get_column_widths(lines)
    yield from (
        f"{_style(line.hash, 'hash', widths)} "
        f"{_style(line.date, 'date', widths)} "
        f"{_style(line.summary, 'summary', widths)} "
        f"{_style(line.line_num, 'line_num', widths, '>')} "
        f"{line.content}\n"
        for line in lines
    )


def _get_column_widths(formatted_lines: list[ParsedLine]) -> dict[str, int]:
    """Calculate the maximum width for each column."""
    return {
        "hash": 7,  # hash is always 7 chars
        "date": 10,  # date is always YYYY-MM-DD
        "summary": max(len(line.summary.strip()) for line in formatted_lines),
        "line_num": len(str(max(line.line_num for line in formatted_lines))),
    }


_COLOURS = {
    "hash": "\033[36m",  # cyan
    "date": "\033[32m",  # green
    "summary": "\033[33m",  # yellow
    "line_num": "\033[35m",  # magenta
    "reset": "\033[0m",  # reset to default
}


def _style(value: object, field: str, widths: dict[str, int], align: str = "<") -> str:
    """Format field as string with the specified colour and determined width."""
    return f"{_COLOURS[field]}{value:{align}{widths[field]}}{_COLOURS['reset']}"


if __name__ == "__main__":
    main()
