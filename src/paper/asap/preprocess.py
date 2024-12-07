"""Run the complete ASAP-Review preprocessing pipeline."""

from pathlib import Path
from typing import Annotated

import typer

from paper.asap.extract import extract_interesting
from paper.asap.filter import filter_ratings
from paper.asap.merge import merge_content_review


def pipeline(
    papers_path: Annotated[
        Path, typer.Argument(help="Path to input directory containing raw ASAP files.")
    ],
    output_path: Annotated[
        Path, typer.Argument(help="Path to output directory for processed files.")
    ],
    max_papers: Annotated[
        int | None, typer.Argument(help="Limit on the number of papers to process.")
    ] = None,
) -> None:
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
    merge_content_review(papers_path, merged_path, max_papers)

    interesting_path = output_path / "asap_extracted.json"
    print(f"\n==== Extracting relevant information from papers -> {interesting_path}")
    extract_interesting(merged_path, interesting_path)

    filtered_path = output_path / "asap_filtered.json"
    print(f"\n==== Removing papers with high variance ratings -> {filtered_path}")
    filter_ratings(interesting_path, filtered_path)

    print(f"\nFinal output file: {filtered_path}")


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
