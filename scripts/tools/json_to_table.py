"""Convert JSON to Markdown table."""

import argparse
import os
import sys
from typing import Any

import orjson


def generate_table(headers: list[str], values: list[list[Any]]) -> str:
    """Generate table from headers and values.

    Args:
        headers: List of strings representing the headers of the table.
        values:
            List of lists of values to be displayed in the table. They will be converted
            to strings before being displayed. Custom formatting must be done before.

    Returns:
        A string representing the table in Markdown format.
    """
    # Calculate the maximum length for each column, considering both headers and the
    # data in values
    max_lengths = [
        max(len(str(row[i])) if i < len(row) else 0 for row in [headers, *values])
        for i in range(len(headers))
    ]

    fmt_parts = [f"{{{i}:<{len}}}" for i, len in enumerate(max_lengths)]
    fmt_string = " | ".join(fmt_parts)

    def format_row(row: list[Any]) -> str:
        return fmt_string.format(*(map(str, row)))

    header_line = format_row(headers)
    separator_line = " | ".join("-" * length for length in max_lengths)
    rows = [format_row(row) for row in values]

    return "\n".join([header_line, separator_line, *rows])


def main() -> None:
    """Convert JSON to pretty table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file",
        type=argparse.FileType(),
        default="-",
        nargs="?",
        help=(
            "The file containing the JSON data to convert to a table. If not provided,"
            " read from stdin."
        ),
    )
    parser.add_argument(
        "--fmt",
        action="append",
        nargs="+",
        metavar=("FORMAT", "COLUMN"),
        help=(
            "Specify the format for one or more columns. Example: --fmt '{:d}' age"
            " --fmt '{:>10}' name email"
        ),
    )
    args = parser.parse_args()

    data = orjson.loads(args.file.read())

    headers = list(data[0].keys())

    fmt_: list[str] = args.fmt
    formats: dict[str, str] = {}
    for fmt_option in fmt_ or []:
        fmt, columns = fmt_option[0], fmt_option[1:]
        for column in columns:
            formats[column] = fmt

    values = [
        [
            formats.get(col, "{}").format(row[col]) if row.get(col) is not None else ""
            for col in headers
        ]
        for row in data
    ]

    # Handle SIGPIPE from head/tail/etc.
    # From https://docs.python.org/3/library/signal.html#note-on-sigpipe
    try:
        print(generate_table(headers, values))
        # flush output here to force SIGPIPE to be triggered
        # while inside this try block.
        sys.stdout.flush()
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE


if __name__ == "__main__":
    main()
