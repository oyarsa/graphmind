"""Run the complete S2ORC preprocessing pipeline. NOTE: may take hours.

Consider running the scripts separately. See the README for more information.
"""

import argparse
import os
from pathlib import Path

from paper_hypergraph.s2orc.download import download_dataset
from paper_hypergraph.s2orc.extract import extract_data
from paper_hypergraph.s2orc.filter import filter_papers


def pipeline(
    api_key: str | None, dataset_path: Path, file_limit: int | None, output_path: Path
) -> None:
    if not api_key:
        api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

    print(f"==== Downloading S2ORC dataset ({file_limit or "all"} files) ====")
    download_dataset("s2orc", dataset_path, api_key, file_limit)

    print(f"\n\n==== Extracting S2ORC dataset to {dataset_path} ====")
    extract_data(dataset_path.glob("*.gz"))

    matched_papers_path = output_path / "papers.json.gz"
    print(f"\n\n==== Get papers matching ACL venues to {matched_papers_path} ====")
    filter_papers(dataset_path, matched_papers_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Semantic Scholar API key. If not provided, uses the"
        " 'SEMANTIC_SCHOLAR_API_KEY' environment variable.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default="data",
        help="Path to save the downloaded dataset",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit the number of files to download",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="output",
        help="Path to save the output files",
    )
    args = parser.parse_args()
    pipeline(
        args.api_key,
        args.dataset_path,
        args.file_limit,
        args.output_path,
    )


if __name__ == "__main__":
    main()
