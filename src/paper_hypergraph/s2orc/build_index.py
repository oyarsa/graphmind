#!/usr/bin/env python3
"""Build an index from multiple gz-compressed JSON files containing paper data.

The index is a JSON file containing an object with fields `title` -> `file name`.
The file names are only the basename for the files, assumed to be under
`input_directory`.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


def process_papers(
    input_directory: Path,
    output_file: Path,
    file_limit: int | None,
) -> None:
    """Process S2ORC papers from multiple compressed JSON files and build an index.

    The input data is the output of the paper_hypergraph.s2orc.extract module.
    The index is title -> file. The title is raw from the 'title' field in the output.
    """

    input_files = list(input_directory.glob("*.json.gz"))
    if not input_files:
        raise ValueError(f"No .json.gz files found in {input_directory}")

    title_index: dict[str, str] = {}

    for file_path in tqdm(input_files[:file_limit]):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as infile:
                data: list[dict[str, Any]] = json.load(infile)

                for paper in data:
                    if title := paper.get("title"):
                        title_index[title] = file_path.name

        except (json.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(title_index, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Path to the directory containing data files.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output index file.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Number of files to process (default: all)",
    )
    args = parser.parse_args()
    process_papers(args.input_directory, args.output_file, args.limit)


if __name__ == "__main__":
    main()
