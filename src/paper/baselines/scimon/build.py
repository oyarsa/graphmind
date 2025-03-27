"""Build the three SciMON graphs (KG, semantic and citations).

This takes two inputs:
- Annotated papers wrapped in prompts (`gpt.PromptResult[gpt.PaperAnnotated]`) from
  `paper.gpt.annotate_paper`.
- PeerRead papers with full S2 reference data (`s2.PaperWithFullS2`) from
  `semantic_scholar.info`.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Annotated

import typer

from paper import embedding as emb
from paper import gpt
from paper import semantic_scholar as s2
from paper.baselines.scimon.graph import Graph
from paper.util import Timer, get_params, render_params, sample, setup_logging
from paper.util.serde import load_data

logger = logging.getLogger("paper.scimon.build")

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
        Path, typer.Option("--ann", help="File with annotated papers.")
    ],
    peerread_file: Annotated[
        Path, typer.Option("--peerread", help="File with PeerRead and references.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory to store the constructed graphs."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = emb.DEFAULT_SENTENCE_MODEL,
    test: Annotated[bool, typer.Option(help="Test graph saving and loading.")] = False,
    num_annotated: Annotated[
        int | None,
        typer.Option(help="Number of annotated papers used for graph (sampled)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for random sample")] = 0,
) -> None:
    """Build the three SciMON graphs (KG, semantic and citations)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    setup_logging()
    params = get_params()
    logger.info(render_params(params))

    logger.debug("Loading data.")

    ann = gpt.PromptResult.unwrap(
        load_data(annotated_file, gpt.PromptResult[gpt.PaperAnnotated])
    )
    ann = sample(ann, num_annotated)

    logger.info("Initialising encoder.")
    encoder = emb.Encoder(model_name)

    peerread_papers = load_data(peerread_file, s2.PaperWithS2Refs)

    with Timer("Building all graphs") as timer_all:
        Graph.build(
            encoder=encoder,
            annotated=ann,
            peerread_papers=peerread_papers,
            output_dir=output_dir,
            metadata=params,
            progress=True,
        )
    logger.info(timer_all)

    if test:
        logger.debug("Testing loading the graph from saved data.")
        _test_load(output_dir)


def _test_load(path: Path) -> None:
    """Test if graph data stored in `path` loads into a valid graph. Tests all three."""
    graph = Graph.load(path)

    kg_result = graph.kg.query("machine learning")
    logger.info("KG: %s", kg_result.model_dump_json(indent=2))

    semantic_result = graph.semantic.query(
        background="We present a family of subgradient methods",
        source="stochastic optimization",
        target="gradient-based learning",
    )
    logger.info("Semantic: %s", semantic_result.model_dump_json(indent=2))

    ctitle = next(iter(graph.citations.title_to_id))
    citation_result = graph.citations.query_title(ctitle, 3)
    logger.info(
        "Citations: %s -> %s", ctitle, citation_result.model_dump_json(indent=2)
    )


if __name__ == "__main__":
    app()
