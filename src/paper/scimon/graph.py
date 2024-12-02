"""Entities in the SciMON baseline reproduction."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict

import paper.scimon.embedding as emb
from paper.scimon import citations, kg, semantic
from paper.scimon.model import PaperAnnotated
from paper.util.serde import load_data

logger = logging.getLogger(__name__)


class GraphData(BaseModel):
    """Serialisation format for `Graph`. Uses each subgraph's data format."""

    model_config = ConfigDict(frozen=True)

    kg: kg.GraphData
    semantic: semantic.GraphData
    citations: citations.Graph
    encoder_model: str
    metadata: str | None = None

    def to_graph(self, encoder: emb.Encoder | None = None) -> Graph:
        """Convert this data to a functioning graph.

        Args:
            encoder: Encoder object to be used by the graph to encode text. If not given,
                will create one from the `encoder_model` name specified in the data.

        Returns:
            A functioning graph, ready to be queried.
        """
        if encoder is None:
            encoder = emb.Encoder(self.encoder_model)
        return Graph(
            kg=self.kg.to_graph(encoder),
            semantic=self.semantic.to_graph(encoder),
            citations=self.citations,
            encoder_model=self.encoder_model,
        )

    @classmethod
    def from_graph(cls, graph: Graph, metadata: str | None = None) -> Self:
        """Convert a graph object to data to be serialised."""
        return cls(
            kg=graph.kg.to_data(),
            semantic=graph.semantic.to_data(),
            citations=graph.citations,
            encoder_model=graph.encoder_model,
            metadata=metadata,
        )


@dataclass(frozen=True, kw_only=True)
class Graph:
    """Collection of KG, Semantic and Citations graph that can be queried together."""

    CITATION_DEFAULT_K: ClassVar[int] = 5

    kg: kg.Graph
    semantic: semantic.Graph
    citations: citations.Graph
    encoder_model: str

    def query_all(
        self, ann: PaperAnnotated, use_kg: bool = False, k: int = CITATION_DEFAULT_K
    ) -> QueryResult:
        """Retrieve terms from the annotated paper using all three graphs.

        KG and Semantic graphs use the `terms` relations. Citations uses the paper `id`.

        Note: each node only appears once across each graph. Citation nodes are paper
        titles, so it doesn't intersect with the other two. Both KG and Semantic nodes
        are relation tails, so it's possible that some appear in both. However, we make
        sure that if a node appears in the KG results, it won't appear in the Semantic
        results.
        """

        if use_kg:
            kg_terms = {
                node
                for relation in ann.terms.relations
                for node in self.kg.query(relation.head).nodes
            }
        else:
            kg_terms: set[str] = set()

        semantic_terms = {
            target
            for relation in ann.terms.relations
            for target in self.semantic.query(
                ann.background, relation.head, relation.tail
            ).targets
        }
        citation_terms = {
            item.title for item in self.citations.query(ann.id, k).citations
        }

        return QueryResult(
            citations=sorted(citation_terms),
            kg=sorted(kg_terms),
            semantic=sorted(semantic_terms - kg_terms),
        )


@dataclass(frozen=True, kw_only=True)
class QueryResult:
    """Query results across graphs, delimited by where they came from."""

    citations: Sequence[str]
    kg: Sequence[str]
    semantic: Sequence[str]


def graph_from_json(file: Path) -> Graph:
    """Read full `Graph` from a JSON `file`.

    See `GraphData` for more info.
    """
    return load_data(file, GraphData, single=True).to_graph()
