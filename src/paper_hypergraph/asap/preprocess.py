"""Run the complete ASAP-Review preprocessing pipeline."""

import argparse
from pathlib import Path

from paper_hypergraph.asap.extract import extract_interesting
from paper_hypergraph.asap.merge import merge_content_review


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Path to input (filtered) JSON file")
    parser.add_argument("output", type=Path, help="Path to output extracted JSON file")
    return parser


def pipeline(papers_path: Path, output_path: Path) -> None:
    merged_path = output_path / "asap_merged.json"
    merge_content_review(papers_path, merged_path)

    interesting_path = output_path / "asap_extracted.json"
    extract_interesting(merged_path, interesting_path)


def main() -> None:
    args = cli_parser().parse_args()
    pipeline(args.input, args.output)
