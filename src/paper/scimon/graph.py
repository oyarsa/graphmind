"""Entities in the SciMON baseline reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict

import paper.scimon.embedding as emb
from paper.gpt.model import PaperTerms
from paper.scimon import citations, kg, semantic
from paper.util.serde import load_data


class GraphData(BaseModel):
    """Serialisation format for `Graph`. Uses each subgraph's data format."""

    model_config = ConfigDict(frozen=True)

    kg: kg.GraphData
    semantic: semantic.GraphData
    citations: citations.Graph
    encoder_model: str

    def to_graph(self, encoder: emb.Encoder | None = None) -> Graph:
        """Convert this data to a functioning graph.

        Args:
            encoder: Encoder object to be used by the graph to encode text. If not given,
                will create one from the one specified in the data.

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
    def from_graph(cls, graph: Graph) -> Self:
        """Convert a graph object to data to be serialised."""
        return cls(
            kg=graph.kg.to_data(),
            semantic=graph.semantic.to_data(),
            citations=graph.citations,
            encoder_model=graph.encoder_model,
        )


class Annotated(Protocol):
    """Protocol for papers with an ID, annotated terms and background."""

    @property
    def terms(self) -> PaperTerms:
        """Extracted terms and relations."""
        ...

    @property
    def id(self) -> str:
        """Unique identifier for the paper."""
        ...

    @property
    def background(self) -> str:
        """Background information from the paper (its problem, task, etc.)."""
        ...


@dataclass(frozen=True, kw_only=True)
class Graph:
    """Collection of KG, Semantic and Citations graph that can be queried together."""

    CITATION_DEFAULT_K: ClassVar[int] = 5

    kg: kg.Graph
    semantic: semantic.Graph
    citations: citations.Graph
    encoder_model: str

    def query_all(self, ann: Annotated, k: int = CITATION_DEFAULT_K) -> list[str]:
        """Retrieve terms from the annotated paper using all three graphs.

        KG and Semantic graphs use the `terms` relations. Citations uses the paper `id`.
        """
        kg_terms = [
            node
            for relation in ann.terms.relations
            for node in self.kg.query(relation.head).nodes
        ]
        semantic_terms = [
            target
            for relation in ann.terms.relations
            for target in self.semantic.query(
                ann.background, relation.head, relation.tail
            ).targets
        ]
        citation_terms = [
            item.title for item in self.citations.query(ann.id, k).citations
        ]
        return sorted(set(kg_terms + semantic_terms + citation_terms))


def graph_from_json(file: Path) -> Graph:
    """Read graph from a JSON `file`. See `GraphData`."""
    return load_data(file, GraphData, single=True).to_graph()
