"""Build a KG graph from extracted terms from papers.

The input is the output of `paper.gpt.annotate_paper`. Since we use the direct output,
of the script, it's wrapped: `run_gpt.PromptResult[annotate_paper.PaperAnnotated]`.

Since we're interested on building the graphs from the relations, we ignore the terms
from `PaperAnnotated.GPTTerms`, and focus only on the relations. The terms are used
for the semantic graph (`paper.scimon.semantic`).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import typer
from openai import BaseModel
from pydantic import ConfigDict

from paper.gpt.annotate_paper import GPTTerms, PaperAnnotated
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
        Path, typer.Argument(help="Output file with the constructed KG graph.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
    query: Annotated[str | None, typer.Option(help="Test query for the graph")] = None,
) -> None:
    setup_logging()
    logger.debug("Starting.")

    logger.debug("Loading data.")
    data = load_data(input_file, PromptResult[PaperAnnotated])
    terms = [x.item.terms for x in data]

    logger.debug("Initialising encoder.")
    with emb.Encoder(model_name) as encoder:
        graph = Graph.from_terms(encoder, terms)

    if query:
        result = graph.query(query)
        logger.info("\nQuery: %s\nResult: %s", query, result)

    output_file.write_text(graph.to_data().model_dump_json(indent=2))


class Graph:
    _nodes: Sequence[str]
    """Nodes are relation heads after processing."""
    _embeddings: emb.Matrix
    """Nodes converted to vectors. Each row corresponds to an element in `_nodes`."""
    _edge_list: Mapping[str, Sequence[str]]
    """Mapping of processed text (relation head) to list of normal text (tails)."""
    _encoder: emb.Encoder
    """Encoder used to convert text nodes to vectors."""

    def __init__(
        self,
        *,
        nodes: Sequence[str],
        embeddings: emb.Matrix,
        edge_list: Mapping[str, Sequence[str]],
        encoder: emb.Encoder,
    ) -> None:
        self._nodes = nodes
        self._embeddings = embeddings
        self._edge_list = edge_list
        self._encoder = encoder

    @classmethod
    def from_terms(cls, encoder: emb.Encoder, terms: Sequence[GPTTerms]) -> Self:
        """Build a graph from a collection of GPTTerms."""
        logger.debug("Building node and edge lists.")

        edge_list: defaultdict[str, list[str]] = defaultdict(list)
        for term in terms:
            for relation in term.relations:
                edge_list[_process_text(relation.head)].append(relation.tail)
        nodes = list(edge_list)

        logger.debug("Encoding nodes.")
        embeddings = encoder.encode_multi(nodes)

        logger.debug("Done.")
        return cls(
            nodes=nodes, embeddings=embeddings, edge_list=edge_list, encoder=encoder
        )

    def query(self, text: str) -> QueryResult:
        """Get neighbours of the best-matching node.

        If the text exists exactly, use that. If not, get the most similar by vector
        similarity. The text is preprocessed first.
        """
        processed = _process_text(text)
        if edges := self._edge_list.get(processed):
            return QueryResult(
                match=processed, nodes=edges, source=QuerySource.EXACT, score=1
            )

        node_embedding = self._encoder.encode(processed)
        similarities = emb.similarities(node_embedding, self._embeddings)
        best = int(similarities.argmax())

        node = self._nodes[best]
        score = similarities[best]
        return QueryResult(
            match=node, nodes=self._edge_list[node], source=QuerySource.SIM, score=score
        )

    def to_data(self) -> GraphData:
        """Convert Graph to a data object."""
        return GraphData(
            embeddings=emb.MatrixData.from_matrix(self._embeddings),
            edge_list=self._edge_list,
            nodes=self._nodes,
        )


class QuerySource(StrEnum):
    EXACT = "exact"
    """Term matches a node in the graph exactly (after pre-processing)."""
    SIM = "sim"
    """Term doesn't match anything, so we take the node with highest similarity."""


class QueryResult(BaseModel):
    """Result of querying the graph. Source depends on how the answer was obtained."""

    model_config = ConfigDict(frozen=True)

    match: str
    nodes: Sequence[str]
    source: QuerySource
    score: float


class GraphData(BaseModel):
    model_config = ConfigDict(frozen=True)

    embeddings: emb.MatrixData
    edge_list: Mapping[str, Sequence[str]]
    nodes: Sequence[str]

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        """Initialise Graph from data object."""
        return Graph(
            nodes=self.nodes,
            embeddings=self.embeddings.to_matrix(),
            edge_list=self.edge_list,
            encoder=encoder,
        )


def _process_text(text: str) -> str:
    """Remove any non-alphabetical characters, except spaces, and casefold."""
    letters_only = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(letters_only.split()).casefold()


if __name__ == "__main__":
    app()
