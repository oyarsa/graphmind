"""Add original paper title to Semantic Scholar API results.

The original version of `paper_hypergraph.s2orc.retrieve_papers_semantic_scholar` did not
include the input title query in the output file, which was necessary to evaluate the
match accuracy. This script adds these titles.

Note: the correct version of the API script is already available, so this is a one-off.
"""

import argparse
import json
from pathlib import Path


def main(titles_file: Path, api_result_file: Path, output_file: Path) -> None:
    titles = json.loads(titles_file.read_text())
    api_result = json.loads(api_result_file.read_text())

    assert len(titles) == len(
        api_result
    ), "Titles and API result must have the same length"

    output = [
        {"title_query": title["title"]} | ({} if result is None else result)
        for title, result in zip(titles, api_result)
    ]

    output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("titles", type=Path, help="Path to ASAP Titles JSON file")
    parser.add_argument(
        "api_result", type=Path, help="Path to Semantic Scholar API results JSON file"
    )
    parser.add_argument("output", type=Path, help="Path to output merged JSON file")
    args = parser.parse_args()
    main(args.titles, args.api_result, args.output)
