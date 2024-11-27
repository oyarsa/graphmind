"""Build a semantic graph from extracted terms and backgrounds from papers.

The input is the output of `paper.gpt.annotate_paper`. Since we use the direct output,
of the script, it's wrapped: `run_gpt.PromptResult[annotate_paper.PaperAnnotated]`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Self

import typer
from openai import BaseModel
from pydantic import ConfigDict

from paper.gpt.annotate_paper import PaperAnnotated
from paper.gpt.run_gpt import PromptResult
from paper.scimon import embedding as emb
from paper.util import setup_logging
from paper.util.serde import load_data

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
    input_file: Annotated[
        Path, typer.Argument(help="Input file with the extracted terms.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Output file with the constructed semantic graph.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    setup_logging()

    logger.debug("Loading data.")
    annotated = PromptResult.unwrap(load_data(input_file, PromptResult[PaperAnnotated]))

    logger.debug("Initialising encoder.")
    with emb.Encoder(model_name) as encoder:
        logger.debug("Constructing graph from annotated paper.")
        graph = Graph.from_annotated(encoder, annotated)

    output_file.write_text(graph.to_data().model_dump_json(indent=2))


class Graph:
    _nodes: Sequence[str]
    """Nodes are base inputs constructed from backgrounds by prompts."""
    _node_to_targets: Mapping[str, Sequence[str]]
    """Mapping of nodes to the paper target terms (relation tails)."""
    _encoder: emb.Encoder
    """Encoder used to convert text nodes to vectors."""
    _embeddings: emb.Matrix
    """Nodes converted to vectors. Each row corresponds to an element in `_nodes`."""

    def __init__(
        self,
        *,
        nodes: Sequence[str],
        embeddings: emb.Matrix,
        node_to_targets: Mapping[str, Sequence[str]],
        encoder: emb.Encoder,
    ) -> None:
        self._nodes = nodes
        self._embeddings = embeddings
        self._node_to_targets = node_to_targets
        self._encoder = encoder

    @classmethod
    def from_annotated(
        cls, encoder: emb.Encoder, annotated: Sequence[PaperAnnotated]
    ) -> Self:
        """Build a semantic graph from annotated papers."""
        logger.debug("Building nodes and edge lists.")

        node_to_targets = {
            base_input: ann.target_terms()
            for ann in annotated
            for relation in ann.terms.relations
            for base_input in _make_base_inputs(
                ann.background, relation.head, relation.tail
            )
        }
        nodes = list(node_to_targets)

        logger.debug("Encoding nodes.")
        embeddings = encoder.encode_multi(nodes)

        logger.debug("Done.")
        return cls(
            nodes=nodes,
            node_to_targets=node_to_targets,
            encoder=encoder,
            embeddings=embeddings,
        )

    def query(self, background: str, source: str, target: str) -> QueryResult:
        """Get paper information from best-matching node by similarity.

        Construct a base input from the information using all `BASE_INPUT_PROMPTS`
        prompts, queries the constructed embeddings and returns the match by highest
        cosine similarity.
        """
        results = (
            self._query_single(base_input)
            for base_input in _make_base_inputs(background, source, target)
        )
        return max(results, key=lambda r: r.score)

    def _query_single(self, base_input: str) -> QueryResult:
        """Query a base input from the embeddings, returning the most similar."""
        embedding = self._encoder.encode(base_input)
        similarities = emb.similarities(embedding, self._embeddings)
        best = int(similarities.argmax())

        node = self._nodes[best]
        score = similarities[best]
        return QueryResult(match=node, targets=self._node_to_targets[node], score=score)

    def to_data(self) -> GraphData:
        """Convert Graph to a data object."""
        return GraphData(
            embeddings=emb.MatrixData.from_matrix(self._embeddings),
            node_to_targets=self._node_to_targets,
            nodes=self._nodes,
            encoder_model=self._encoder.model_name,
        )


class QueryResult(BaseModel):
    """Result of querying the semantic graph."""

    model_config = ConfigDict(frozen=True)

    match: str
    targets: Sequence[str]
    score: float


class GraphData(BaseModel):
    model_config = ConfigDict(frozen=True)

    embeddings: emb.MatrixData
    node_to_targets: Mapping[str, Sequence[str]]
    nodes: Sequence[str]
    encoder_model: str

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        """Initialise Graph from data object.

        Raises:
            ValueError: `encoder` model is different from the one that generated the
            graph.
        """
        if encoder.model_name != self.encoder_model:
            raise ValueError(
                f"Incompatible encoder. Expected '{self.encoder_model}', got"
                f" '{encoder.model_name}'"
            )
        return Graph(
            nodes=self.nodes,
            embeddings=self.embeddings.to_matrix(),
            node_to_targets=self.node_to_targets,
            encoder=encoder,
        )


_BASE_INPUT_PROMPTS = [
    "{source} is used for {target}",
    "{target} is done by using {source}",
]
"""Prompts to generate base inputs for semantic nodes."""


def _make_base_inputs(background: str, source: str, target: str) -> list[str]:
    """Create base inputs from background, source and target terms. Uses all prompts.

    See `BASE_INPUT_PROMPTS` for the prompts.

    From the paper:
        Given the background context B with a seed term v and problem/motivation M, we
        construct a base input b: a concatenation of M with a prompt P belonging to one
        of two templates: “v is used for p” or “v is done by using p”, where p is one of
        Task/Method/Material/Metric. In short, b := P ⊕ context:M.
    """
    return [
        f"{prompt.format(source=source, target=target)}. Context: {background}"
        for prompt in _BASE_INPUT_PROMPTS
    ]


if __name__ == "__main__":
    app()
