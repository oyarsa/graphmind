"""Build the three SciMON graphs (KG, semantic and citations) as a single structure.

The stored graph needs to be converted to a real one in memory because of how the
embeddings are stored.

This takes two inputs:
- Annotated papers wrapped in prompts (gpt.PromptResult[gpt.PaperAnnotated]) from
  `paper.gpt.annotate_paper`.
- ASAP papers with full S2 reference data (s2.ASAPWithFullS2) from
  `semantic_scholar.info`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

import paper.semantic_scholar as s2
from paper import embedding as emb
from paper import gpt
from paper.scimon import citations, kg, semantic
from paper.scimon.graph import Graph, GraphData
from paper.util import Timer, display_params, setup_logging
from paper.util.serde import load_data, save_data

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
    output_file: Annotated[
        Path, typer.Option("--output", help="Output file with the constructed graphs.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
    test: Annotated[bool, typer.Option(help="Test graph saving and loading.")] = False,
) -> None:
    """Build the three SciMON graphs (KG, semantic and citations) as a single structure."""
    setup_logging()
    params = display_params()
    logger.info(params)

    logger.debug("Loading data.")

    ann = gpt.PromptResult.unwrap(
        load_data(annotated_file, gpt.PromptResult[gpt.PaperAnnotated])
    )
    asap_papers = load_data(asap_file, s2.ASAPWithFullS2)
    terms = [x.terms for x in ann]

    logger.info("Initialising encoder.")
    encoder = emb.Encoder(model_name)

    logger.info("Building graphs")

    logger.info("Building Semantic: %d annotations", len(ann))
    with Timer("Semantic") as timer_semantic:
        semantic_graph = semantic.Graph.from_annotated(encoder, ann, progress=True)
    logger.info(timer_semantic)

    logger.info("Building KG: %d terms", len(terms))
    with Timer("KG") as timer_kg:
        kg_graph = kg.Graph.from_terms(encoder, terms, progress=True)
    logger.info(timer_kg)

    logger.info("Building Citation: %d papers", len(asap_papers))
    with Timer("Citation") as timer_citation:
        citation_graph = citations.Graph.from_papers(encoder, asap_papers)
    logger.info(timer_citation)

    logger.info("Saving graphs")
    graph = Graph(
        kg=kg_graph,
        semantic=semantic_graph,
        citations=citation_graph,
        encoder_model=model_name,
    )
    graph_data = GraphData.from_graph(graph, metadata=params)
    save_data(output_file, graph_data)

    if test:
        logger.debug("Testing loading the graph from saved data.")
        _test_load(output_file)


def _test_load(path: Path) -> None:
    """Test if graph data stored in `path` loads into a valid graph. Tests all three."""
    data = load_data(path, GraphData, single=True)

    graph = data.to_graph()

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
