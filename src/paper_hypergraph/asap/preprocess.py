"""Run the complete ASAP-Review preprocessing pipeline."""

import argparse
from pathlib import Path

from paper_hypergraph.asap.extract import extract_interesting
from paper_hypergraph.asap.filter import filter_ratings
from paper_hypergraph.asap.merge import merge_content_review


def pipeline(papers_path: Path, output_path: Path, context_sentences: int) -> None:
    """Run the complete ASAP-Review preprocessing pipeline.

    Steps:
    1. Read all papers from the input paper directory and merge them a single JSON file.
       Keeps the content and reviews of papers that have ratings.
    2. Extract the interesting information from the merged JSON file.
    3. Filter the extracted information to remove irrelevant papers.

    Writes the output JSON files to the output directory. Intermediate files are also
    saved there.
    """
    merged_path = output_path / "asap_merged.json"
    print(f"==== Merging data from multiple files -> {merged_path}")
    merge_content_review(papers_path, merged_path)

    interesting_path = output_path / "asap_extracted.json"
    print(f"\n==== Extracting relevant information from papers -> {interesting_path}")
    extract_interesting(merged_path, interesting_path, context_sentences)

    filtered_path = output_path / "asap_filtered.json"
    print(f"\n==== Removing papers with high variance ratings -> {filtered_path}")
    filter_ratings(interesting_path, filtered_path)

    print(f"\nFinal output file: {filtered_path}")


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="Path to input directory containing raw ASAP files"
    )
    parser.add_argument(
        "output", type=Path, help="Path to output directory for processed files"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=1,
        help="Maximum number of sentences to expand the context (before and after)",
    )
    return parser


def main() -> None:
    args = cli_parser().parse_args()
    pipeline(args.input, args.output, args.context_sentences)


if __name__ == "__main__":
    main()
