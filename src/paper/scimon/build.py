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

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, TypeAdapter

import paper.external_data.semantic_scholar.model as s2
from paper.gpt.annotate_paper import PaperAnnotated
from paper.gpt.run_gpt import PromptResult
from paper.scimon import citations, kg, semantic
from paper.scimon import embedding as emb
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

    graph_data = SciMONData(
        kg=kg_graph.to_data(),
        semantic=semantic_graph.to_data(),
        citations=citation_graph,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph_data.model_dump_json(indent=2))

    logger.debug("Testing loading the graph from saved data.")
    _test_load(output, model_name)


class SciMONData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kg: kg.GraphData
    semantic: semantic.GraphData
    citations: citations.Graph

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        return Graph(
            kg=self.kg.to_graph(encoder),
            semantic=self.semantic.to_graph(encoder),
            citations=self.citations,
        )


@dataclass(frozen=True, kw_only=True)
class Graph:
    kg: kg.Graph
    semantic: semantic.Graph
    citations: citations.Graph


def _test_load(path: Path, model: str) -> None:
    """Test if graph data stored in `path` loads into a valid graph. Tests all three."""
    data = SciMONData.model_validate_json(path.read_text())

    with emb.Encoder(model) as encoder:
        graph = data.to_graph(encoder)

        kg_result = graph.kg.query("machine learning")
        logger.info("KG: %s", kg_result.model_dump_json(indent=2))

        semantic_result = graph.semantic.query(
            "We present a family of subgradient methods",
            "stochastic optimization",
            "gradient-based learning",
        )
        semantic_view = {
            "match": semantic_result.match,
            "paper_name": semantic_result.paper.paper.title,
            "score": semantic_result.score,
            "terms": semantic_result.paper.terms.model_dump(),
            "background": semantic_result.paper.background,
            "target": semantic_result.paper.target,
        }
        logger.info("Semantic: %s", json.dumps(semantic_view, indent=2))

        ctitle, cnode = next(iter(graph.citations.title_to_id.items()))
        citation_result = graph.citations.query(cnode, 3)
        logger.info(
            "Citations: %s -> %s",
            ctitle,
            TypeAdapter(Sequence[citations.Citation])
            .dump_json(citation_result, indent=2)
            .decode(),
        )


if __name__ == "__main__":
    app()
