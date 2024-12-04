"""Create PETER graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from paper import embedding as emb
from paper import gpt
from paper.peter import citations
from paper.util import display_params, setup_logging
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Set up logging for all commands."""
    setup_logging()


@app.command(name="citations", help="Create citations graph", no_args_is_help=True)
def citations_(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="File with ASAP papers with references with full S2 data and classified"
            " contexts."
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(
            help="File with ASAP papers with top K references with full S2 data."
        ),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    logger.debug("Loading classified papers.")
    asap_papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithContextClassfied])
    )

    logger.debug("Loading encoder.")
    encoder = emb.Encoder(model_name)
    logger.debug("Creating graph.")
    graph = citations.Graph.from_papers(encoder, asap_papers, progress=True)

    logger.debug("Saving graph.")
    save_data(output_file, graph)


if __name__ == "__main__":
    app()
