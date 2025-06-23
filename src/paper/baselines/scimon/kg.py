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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Self

import typer

from paper import embedding as emb
from paper import gpt
from paper.types import Immutable
from paper.util import setup_logging
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
    input_file: Annotated[
        Path, typer.Argument(help="Input file with the extracted terms.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Output file with the constructed KG graph.")
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = emb.DEFAULT_SENTENCE_MODEL,
    query: Annotated[str | None, typer.Option(help="Test query for the graph")] = None,
) -> None:
    """Build a KG graph from extracted terms from papers."""
    setup_logging()
    logger.debug("Starting.")

    logger.debug("Loading data.")
    data = load_data(input_file, gpt.PromptResult[gpt.PaperAnnotated])
    terms = [x.item.terms for x in data]

    logger.debug("Initialising encoder.")
    encoder = emb.Encoder(model_name)
    graph = Graph.from_terms(encoder, terms)

    if query:
        result = graph.query(query)
        logger.info("\nQuery: %s\nResult: %s", query, result)

    save_data(output_file, graph.to_data())


class Graph:
    """Conventional graph created by used-for relations.

    Querying the graph can be made by exact match between nodes in the graph and a query
    term. Failing that, we retrieve the closest match by semantic similarity.
    """

    _nodes: Sequence[str]
    """Nodes are relation heads after processing."""
    _embeddings: emb.Matrix
    """Nodes converted to vectors. Each row corresponds to an element in `_nodes`."""
    _head_to_tails: Mapping[str, Sequence[str]]
    """Mapping of processed text (relation head) to list of normal text (tails)."""
    _encoder: emb.Encoder
    """Encoder used to convert text nodes to vectors."""

    def __init__(
        self,
        *,
        nodes: Sequence[str],
        embeddings: emb.Matrix,
        head_to_tails: Mapping[str, Sequence[str]],
        encoder: emb.Encoder,
    ) -> None:
        self._nodes = nodes
        self._embeddings = embeddings
        self._head_to_tails = head_to_tails
        self._encoder = encoder

    @classmethod
    def from_terms(
        cls,
        encoder: emb.Encoder,
        terms: Iterable[gpt.PaperTerms],
        *,
        progress: bool = False,
    ) -> Self:
        """Build a graph from a collection of annotated `PaperTerms`."""
        logger.debug("Building node and edge lists.")

        head_to_tails: defaultdict[str, list[str]] = defaultdict(list)
        for term in terms:
            for relation in term.relations:
                head_to_tails[_process_text(relation.head)].append(relation.tail)
        nodes = list(head_to_tails)

        logger.debug("Encoding nodes.")
        embeddings = encoder.batch_encode(nodes, progress=progress)

        logger.debug("Done.")
        return cls(
            nodes=nodes,
            embeddings=embeddings,
            head_to_tails=head_to_tails,
            encoder=encoder,
        )

    def query(self, text: str) -> QueryResult:
        """Get neighbours of the node matching `text`.

        If there isn't a match, returns an empty list.
        """
        processed = _process_text(text)
        if neighbours := self._head_to_tails.get(processed):
            return QueryResult(match=processed, nodes=neighbours)
        return QueryResult(match=processed, nodes=[])

    def to_data(self) -> GraphData:
        """Convert KG Graph to a data object."""
        return GraphData(
            embeddings=emb.MatrixData.from_matrix(self._embeddings),
            head_to_tails=self._head_to_tails,
            nodes=self._nodes,
            encoder_model=self._encoder.model_name,
        )


class QueryResult(Immutable):
    """Result of querying the graph. Source depends on how the answer was obtained."""

    match: str
    nodes: Sequence[str]


class GraphData(Immutable):
    """Serialisation format for the KG graph.

    The KG graph includes an encoder and embedding matrices. We convert the latter to a
    text-based representation. We don't store the encoder, only its model name.
    """

    embeddings: emb.MatrixData
    head_to_tails: Mapping[str, Sequence[str]]
    nodes: Sequence[str]
    encoder_model: str

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        """Initialise KG Graph from data object.

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
            head_to_tails=self.head_to_tails,
            encoder=encoder,
        )


def _process_text(text: str) -> str:
    """Remove any non-alphabetical characters, except spaces, and casefold."""
    letters_only = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(letters_only.split()).casefold()


if __name__ == "__main__":
    app()
