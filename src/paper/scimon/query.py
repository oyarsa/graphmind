"""Query all terms for PeerRead-annotated papers and save the result as JSON.

The inputs are:
- Annotated PeerRead papers from `gpt.annotate_paper`.
- The SciMON graph created from annotated S2 papers (also `gpt.annotate_paper`) via
  `scimon.build`.
"""

from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from paper import gpt
from paper.scimon.graph import AnnotatedGraphResult, graph_from_json
from paper.util.serde import load_data, save_data

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
    graph_file: Annotated[
        Path,
        typer.Option("--graph", help="JSON file containing the SciMON graphs."),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path to the output file with annotated papers and their graph results.",
        ),
    ],
) -> None:
    """Query all annotatedd papers in the graph."""
    anns = gpt.PromptResult.unwrap(
        load_data(annotated_file, gpt.PromptResult[gpt.PeerReadAnnotated])
    )
    graph = graph_from_json(graph_file)

    ann_result = [
        AnnotatedGraphResult(ann=ann, result=graph.query_all(ann))
        for ann in tqdm(anns, desc="Querying graph with annotated papers")
    ]

    save_data(output_file, ann_result)


if __name__ == "__main__":
    app()
