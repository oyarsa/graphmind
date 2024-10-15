"""Remove papers whose ratings have too much variance.

This is determined by the difference between the minimum and maximum ratings. If it's
greater than 3, the paper is removed from the dataset.
"""

import argparse
from pathlib import Path

from paper_hypergraph.asap.model import ASAPDatasetAdapter, Paper


def _keep_paper(paper: Paper) -> bool:
    """Paper is kept if the difference between min and max ratings is <= 3."""
    return max(paper.ratings) - min(paper.ratings) <= 3


def filter_ratings(input_file: Path, output_file: Path) -> None:
    """Remove papers whose ratings have too much variance from the dataset.

    The input file is the output of `paper_hypergraph.asap.extract`. The output has
    the same format as the input.
    """
    data = ASAPDatasetAdapter.validate_json(input_file.read_text())
    output = [p for p in data if _keep_paper(p)]

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    output_file.write_bytes(ASAPDatasetAdapter.dump_json(output))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Path to input (extracted) JSON file")
    parser.add_argument("output", type=Path, help="Path to output filtered JSON file")
    args = parser.parse_args()
    filter_ratings(args.input, args.output)


if __name__ == "__main__":
    main()
