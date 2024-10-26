"""Calculate total and average number of unique reference contexts per paper in ASAP.

Also calculates the number of words in the contexts and estimated number of GPT tokens.

Uses the output of `paper.asap.merge` as input.
"""

import argparse
import json
from pathlib import Path


def main(input_file: Path) -> None:
    counts_unique: list[int] = []
    all_unique: set[str] = set()

    data = json.loads(input_file.read_text())
    for entry in data:
        unique_contexts: set[str] = {
            r["context"].strip() for r in entry["paper"]["referenceMentions"]
        }
        all_unique.update(unique_contexts)
        counts_unique.append(len(unique_contexts))

    print(f"{len(all_unique):,} total unique mentions")
    print(f"{sum(counts_unique) / len(counts_unique):.2f} average unique mentions")

    num_tokens = sum(len(context.split()) for context in all_unique)
    print(f"{num_tokens:,} total words")
    print(f"{round(num_tokens * 1.5):,} estimated GPT tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (ASAP merged) to calculate statistics from.",
    )
    args = parser.parse_args()
    main(args.input)
