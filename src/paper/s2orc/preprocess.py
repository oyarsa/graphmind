"""Run the complete S2ORC preprocessing pipeline. NOTE: may take hours.

Consider running the scripts separately. See the README for more information.
"""

import os
from pathlib import Path

import dotenv

from paper.s2orc.download import download_dataset
from paper.s2orc.extract import extract_data
from paper.s2orc.filter import filter_papers
from paper.util import HelpOnErrorArgumentParser


def pipeline(
    processed_dir: Path,
    output_path: Path,
    api_key: str | None,
    dataset_path: Path,
    file_limit: int | None,
) -> None:
    """Run the complete S2ORC preprocessing pipeline.

    Steps:
    1. Download the entire S2ORC dataset to disk (all .gz files that contain JSON lines).
    2. Extract the .gz files to JSON.GZ files. Only keep those that contain the title
       and annotations (e.g. abstract, venue, text).
    3. Filter the papers to only those that match the ACL venues.
    """

    dotenv.load_dotenv()
    if not api_key:
        api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

    print(
        f"==== Downloading S2ORC dataset ({file_limit or "all"} files) -> {dataset_path}"
    )
    download_dataset("s2orc", dataset_path, api_key, file_limit)

    print(f"\n\n==== Extracting S2ORC dataset -> {dataset_path}")
    extract_data(dataset_path.glob("*.gz"), processed_dir)

    matched_papers_path = output_path / "s2orc_papers.json.gz"
    print(f"\n==== Get papers matching ACL venues -> {matched_papers_path}")
    filter_papers(dataset_path, matched_papers_path)

    print(f"\nFinal output file: {matched_papers_path}")


def cli_parser() -> HelpOnErrorArgumentParser:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to save the downloaded dataset",
    )
    parser.add_argument(
        "processed_path",
        type=Path,
        help="Path to save the S2 extracted files (JSON.GZ)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save the output (processed and filtered - ACL only) files",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Semantic Scholar API key. If not provided, uses the"
        " 'SEMANTIC_SCHOLAR_API_KEY' environment variable.",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit the number of files to download. If not provided, download all.",
    )
    return parser


def main() -> None:
    args = cli_parser().parse_args()
    pipeline(
        args.processed_path,
        args.output_path,
        args.api_key,
        args.dataset_path,
        args.file_limit,
    )


if __name__ == "__main__":
    main()
