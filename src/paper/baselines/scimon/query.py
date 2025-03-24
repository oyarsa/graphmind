"""Query all terms for PeerRead-annotated papers and save the result as JSON.

The inputs are:
- Annotated PeerRead papers from `gpt.annotate_paper`.
- The SciMON graph created from annotated S2 papers (also `gpt.annotate_paper`) via
  `scimon.build`.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from paper import gpt
from paper.baselines.scimon.graph import AnnotatedGraphResult, Graph
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    annotated_file: Annotated[
        Path,
        typer.Option(
            "--ann-peer",
            help="JSON file containing the annotated PeerRead papers data.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path to the output file with annotated papers and their graph results.",
        ),
    ],
    graph_dir: Annotated[
        Path,
        typer.Option(
            "--graph-dir", help="Directory containing the SciMON graph files."
        ),
    ],
) -> None:
    """Query all annotated papers in the graph."""
    anns = gpt.PromptResult.unwrap(
        load_data(annotated_file, gpt.PromptResult[gpt.PeerReadAnnotated])
    )
    graph = Graph.load(graph_dir)

    ann_result = [
        AnnotatedGraphResult(ann=ann, result=graph.query_all(ann))
        for ann in tqdm(anns, desc="Querying graph with annotated papers")
    ]

    save_data(output_file, ann_result)


if __name__ == "__main__":
    app()
