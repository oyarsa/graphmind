#!/usr/bin/env python3
"""Extract unique venues from gzipped JSON files."""

import argparse
import gzip
import json
import sys
from pathlib import Path

from tqdm import tqdm


def main(directory: Path, output_file: Path) -> None:
    files = list(directory.rglob("*.json.gz"))
    venues: set[str] = set()

    for file_path in tqdm(files):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                venues.update(
                    venue.casefold().replace("\n", " ")
                    for item in data
                    if (venue := item.get("venue", "").strip())
                )
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(sorted(venues)) + "\n", encoding="utf-8")

    print(len(venues), "venues found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directory", type=Path, help="Path to the directory containing data files."
    )
    parser.add_argument("output_file", type=Path, help="File path to save the output.")
    args = parser.parse_args()
    main(args.directory, args.output_file)
