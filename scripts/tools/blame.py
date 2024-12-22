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


def _style(value: object, field: str, widths: dict[str, int], align: str = "<") -> str:
    """Format field as string with the specified colour and determined width."""
    return f"{COLOURS[field]}{value:{align}{widths[field]}}{COLOURS['reset']}"


def main() -> None:
    """Format git blame output nicely, including other CLI arguments."""
    cmd = ["git", "blame", "--porcelain"] + sys.argv[1:]
    result = subprocess.run(cmd, capture_output=True, check=True, text=True)

    current_commit = CurrentCommit()
    formatted_lines: list[FormattedLine] = []
    commits_cache: dict[str, CurrentCommit] = {}  # Cache commit metadata

    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        if line.startswith("\t"):
            # This is the actual file line
            content = line[1:]  # Remove the tab
            if current_commit.is_complete():
                # We know these aren't None because is_complete() returned True
                assert current_commit.hash is not None
                assert current_commit.date is not None
                assert current_commit.summary is not None
                assert current_commit.line_num is not None

                formatted_lines.append(
                    FormattedLine(
                        hash=current_commit.hash[:7],
                        date=current_commit.date,
                        summary=current_commit.summary,
                        line_num=current_commit.line_num,
                        content=content,
                    )
                )
        else:
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 40:  # Full SHA-1 hash is 40 chars
                # Check if we've seen this commit before
                if parts[0] in commits_cache:
                    cached = commits_cache[parts[0]]
                    # We know these aren't None because we only cache complete commits
                    assert cached.hash is not None
                    assert cached.date is not None
                    assert cached.summary is not None

                    current_commit = CurrentCommit(
                        hash=cached.hash,
                        date=cached.date,
                        summary=cached.summary,
                        line_num=int(parts[1]),
                    )
                else:
                    # New commit, need to parse its metadata
                    current_commit = CurrentCommit(
                        hash=parts[0], line_num=int(parts[1])
                    )
                    # Parse subsequent lines for metadata until we hit a tab or another hash
                    j = i + 1
                    while j < len(lines) and not (
                        lines[j].startswith("\t")
                        or (
                            len(lines[j].split()) >= 2
                            and len(lines[j].split()[0]) == 40
                        )
                    ):
                        metadata = lines[j]
                        if metadata.startswith("author-time "):
                            timestamp = int(metadata.split()[1])
                            current_commit.date = dt.datetime.fromtimestamp(
                                timestamp, tz=dt.UTC
                            ).strftime("%Y-%m-%d")
                        elif metadata.startswith("summary "):
                            current_commit.summary = metadata[8:]
                        j += 1
                    i = j - 1  # Move main loop to where we ended
                    # Cache this commit's metadata
                    if current_commit.is_complete():
                        commits_cache[parts[0]] = current_commit
        i += 1

    if not formatted_lines:
        print("No lines were formatted!", file=sys.stderr)
        return

    widths = _get_column_widths(formatted_lines)
    output: list[str] = []
    for line in formatted_lines:
        output.append(
            f"{_style(line.hash, 'hash', widths)} "
            f"{_style(line.date, 'date', widths)} "
            f"{_style(line.summary, 'summary', widths)} "
            f"{_style(line.line_num, 'line_num', widths, '>')} "
            f"{line.content}\n"
        )
    click.echo_via_pager(output)


if __name__ == "__main__":
    main()
