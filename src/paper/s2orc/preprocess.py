"""Run the complete S2ORC preprocessing pipeline. NOTE: may take hours.

Consider running the scripts separately. See the README for more information.
"""

from pathlib import Path
from typing import Annotated

import typer

from paper.s2orc.datasets import download_dataset
from paper.s2orc.extract import extract_data
from paper.s2orc.filter import filter_papers


def pipeline(
    processed_dir: Annotated[
        Path, typer.Argument(help="Path to save the downloaded dataset.")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Path to save the S2 extracted files (JSON.GZ).")
    ],
    dataset_path: Annotated[
        Path,
        typer.Argument(
            help="Path to save the output (processed and filtered - ACL only) files"
        ),
    ],
    file_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit the number of files to download. If not provided, download all."
        ),
    ] = None,
) -> None:
    """Run the complete S2ORC preprocessing pipeline.

    Steps:
    1. Download the entire S2ORC dataset to disk (all .gz files that contain JSON lines).
    2. Extract the .gz files to JSON.GZ files. Only keep those that contain the title
       and annotations (e.g. abstract, venue, text).
    3. Filter the papers to only those that match the ACL venues.
    """

    print(
        f"==== Downloading S2ORC dataset ({file_limit or "all"} files) -> {dataset_path}"
    )
    download_dataset("s2orc", dataset_path, file_limit)

    print(f"\n\n==== Extracting S2ORC dataset -> {dataset_path}")
    extract_data(dataset_path.glob("*.gz"), processed_dir)

    matched_papers_path = output_path / "s2orc_papers.json.gz"
    print(f"\n==== Get papers matching ACL venues -> {matched_papers_path}")
    filter_papers(dataset_path, matched_papers_path)

    print(f"\nFinal output file: {matched_papers_path}")


if __name__ == "__main__":
    # Defined here so that `paper.pipeline` can use the `pipeline` function directly.
    _app = typer.Typer(
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
        no_args_is_help=True,
    )
    _app.command(help=__doc__, no_args_is_help=True)(pipeline)
    _app()
