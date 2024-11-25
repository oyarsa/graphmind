"""Build the three SciMON graphs (KG, semantic and citations) as a single structure.

The stored graph needs to be converted to a real one in memory because of how the
embeddings are stored.

This takes two inputs:
- Annotated papers wrapped in prompts (gpt.PromptResult[PaperAnnotated]) from
  `paper.gpt.annotate_paper`.
- ASAP papers with full S2 reference data (s2.ASAPWithFullS2) from
  `external_data.semantic_scholar.info`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

import paper.external_data.semantic_scholar.model as s2
from paper.gpt.annotate_paper import PaperAnnotated
from paper.gpt.run_gpt import PromptResult
from paper.scimon import citations, kg, semantic
from paper.scimon import embedding as emb
from paper.scimon.model import GraphData
from paper.util import display_params, setup_logging
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
    asap_file: Annotated[
        Path, typer.Option("--asap", help="File with ASAP and references.")
    ],
    output: Annotated[
        Path, typer.Option(help="Output file with the constructed graphs.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    setup_logging()
    logger.info(display_params())

    logger.debug("Loading data.")

    data = load_data(annotated_file, PromptResult[PaperAnnotated])
    asap_papers = load_data(asap_file, s2.ASAPWithFullS2)
    ann = [x.item for x in data]
    terms = [x.terms for x in ann]

    logger.debug("Initialising encoder.")
    with emb.Encoder(model_name) as encoder:
        logger.debug("Building graphs")
        kg_graph = kg.Graph.from_terms(encoder, terms)
        semantic_graph = semantic.Graph.from_annotated(encoder, ann)
        citation_graph = citations.Graph.from_papers(encoder, asap_papers)

    graph_data = GraphData(
        kg=kg_graph.to_data(),
        semantic=semantic_graph.to_data(),
        citations=citation_graph,
        encoder_model=model_name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph_data.model_dump_json(indent=2))

    logger.debug("Testing loading the graph from saved data.")
    _test_load(output, model_name)


def _test_load(path: Path, model: str) -> None:
    """Test if graph data stored in `path` loads into a valid graph. Tests all three."""
    data = GraphData.model_validate_json(path.read_text())

    with emb.Encoder(model) as encoder:
        graph = data.to_graph(encoder)

        kg_result = graph.kg.query("machine learning")
        logger.info("KG: %s", kg_result.model_dump_json(indent=2))

        semantic_result = graph.semantic.query(
            background="We present a family of subgradient methods",
            source="stochastic optimization",
            target="gradient-based learning",
        )
        logger.info("Semantic: %s", semantic_result.model_dump_json(indent=2))

        ctitle = "BLOCK-DIAGONAL HESSIAN-FREE OPTIMIZATION"
        citation_result = graph.citations.query_title(ctitle, 3)
        logger.info(
            "Citations: %s -> %s", ctitle, citation_result.model_dump_json(indent=2)
        )


if __name__ == "__main__":
    app()
