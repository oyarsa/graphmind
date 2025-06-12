"""Fetch conference paper data from the OpenReview API and download LaTeX from arXiv.

The process for retrieving the whole data is running the subcommands in this order:
- reviews
- latex
- parse
- preprocess

To download all the relevant conference information, use:
- reviews-all
- latex-all
- parse-all
- preprocess

`parse` and `parse-all` require pandoc to convert LaTeX to Markdown:
https://pandoc.org/installing.html.

Use the same output/data directory for all of them.
"""

# pyright: basic
import logging
from pathlib import Path
from typing import Annotated

import typer

from paper.orc.arxiv import latex, latex_all
from paper.orc.download import reviews, reviews_all, reviews_from_titles
from paper.orc.latex_parser import parse, parse_all
from paper.orc.preprocess import preprocess
from paper.util import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

# Register commands
app.command()(latex)
app.command()(latex_all)
app.command()(reviews)
app.command()(reviews_all)
app.command()(reviews_from_titles)
app.command()(parse)
app.command()(parse_all)
app.command()(preprocess)


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


@app.command(no_args_is_help=True, name="all")
def all_(
    data_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to save the output JSON file.")
    ],
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Run full ORC pipeline.

    - Download reviews from the OpenReview API.
    - Download LaTeX code from arXiv.
    - Transform the LaTeX code into Markdown and JSON with citations.
    - Merge and transform into a single file.

    These are the conferences used:

    - ICLR 2022, 2023, 2024, 2025
    - NeurIPS 2022, 2023, 2024

    ICLR 2022 and 2023 have "technical" and "empirical" novelty as ratings. We use the
    higher one for the target. For the rest, we use the "contribution" rating.

    These are all numerical ratings from 1 to 4. We also convert them to binary, with
    1-2 being not novel and 3-4 being novel.
    """
    reviews_all(data_dir, query_arxiv=False)
    latex_all(data_dir, max_papers=num_papers, clean_run=clean_run)
    parse_all(data_dir, max_items=num_papers, clean=clean_run)
    preprocess(data_dir, output_file, num_papers=num_papers)


if __name__ == "__main__":
    app()
