"""Add paper query information to Crossref "best" file.

The original version of the Crossref script only added the query stuff to the "full"
file, but it's interesting to have it in the "best" file too to compare the best output
with the original papers.

This is done by going through the "full" file and doing the same fuzzy comparison the
original script did.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from paper_hypergraph.s2orc.query_cr import get_best_paper


def download(
    input_file: Path,
    output_file: Path,
    fuzz_threshold: int,
) -> None:
    input_papers = json.loads(input_file.read_text())

    output_best: list[dict[str, Any]] = []

    for paper_full in tqdm(input_papers):
        if (papers := paper_full.get("message", {}).get("items")) and (
            best := get_best_paper(paper_full["query"]["title"], papers, fuzz_threshold)
        ):
            output_best.append(
                {
                    "query": {
                        "title": paper_full["query"]["title"],
                        "author": paper_full["query"]["author"],
                    }
                }
                | best
            )

    print(f"Before: {len(input_papers)} papers")
    print(
        f"After : {len(output_best)} papers ({len(output_best) / len(input_papers):.2%})"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_best))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=Path, help="File containing paper titles, one per line"
    )
    parser.add_argument(
        "output_file", type=Path, help="Directory to save the downloaded information"
    )
    parser.add_argument(
        "--ratio", type=int, default=30, help="Minimum ratio for fuzzy matching"
    )
    args = parser.parse_args()

    while True:
        try:
            download(args.input_file, args.output_file, args.ratio)
            break  # If _download completes without interruption, exit the loop
        except KeyboardInterrupt as e:
            choice = input("\n\nCtrl+C detected. Do you really want to exit? (y/n): ")
            if choice.lower() == "y":
                print(e)
                sys.exit()
            else:
                # The loop will continue, restarting _download
                print("Continuing...\n")


if __name__ == "__main__":
    main()
