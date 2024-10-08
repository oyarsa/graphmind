#!/usr/bin/env python3
"""Combine multiple .json.gz files into one."""

import argparse
import gzip
import json
import sys
from pathlib import Path

from tqdm import tqdm


def _combine_papers(files: list[Path]) -> list[dict[str, str]]:
    """Extract papers from ACL-related venues."""
    papers: list[dict[str, str]] = []

    for file_path in tqdm(files, desc="Extracting papers"):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                for paper in data:
                    papers.append(paper | {"source": file_path.stem})
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return papers


def filter_papers(input_directory: Path, output_file: Path) -> None:
    """Keep only ACL-related papers from processed JSON.GZ files.

    The input data is the output of the paper_hypergraph.s2orc.extract module.
    """
    input_files = list(input_directory.rglob("*.json.gz"))
    if not input_files:
        raise ValueError(f"No .json.gz files found in {input_directory}")

    papers = _combine_papers(input_files)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wt") as outfile:
        json.dump(papers, outfile, indent=2)

    print(f"{len(papers)} papers extracted and saved to {output_file}")


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
        "output_file", type=Path, help="Path to save the output .json.gz file."
    )
    args = parser.parse_args()
    filter_papers(args.input_directory, args.output_file)


if __name__ == "__main__":
    main()
