"""Entities in the SciMON baseline reproduction."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

from paper import embedding as emb
from paper import gpt
from paper import semantic_scholar as s2
from paper.baselines.scimon import citations, kg, semantic
from paper.types import Immutable
from paper.util.serde import Record, load_data_single, save_data

logger = logging.getLogger(__name__)


class MetadataModel(Immutable):
    """Metadata for the graph."""

    encoder_model: str
    metadata: dict[str, Any] | None = None


class GraphData(Immutable):
    """Serialisation format for `Graph`. Uses each subgraph's data format."""

    kg: kg.GraphData
    semantic: semantic.GraphData
    citations: citations.Graph
    encoder_model: str
    metadata: dict[str, Any] | None = None

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
    def from_graph(cls, graph: Graph, metadata: dict[str, Any] | None = None) -> Self:
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
    KG_FILENAME: ClassVar[str] = "kg_graph.json"
    SEMANTIC_FILENAME: ClassVar[str] = "semantic_graph.json"
    CITATIONS_FILENAME: ClassVar[str] = "citation_graph.json"
    METADATA_FILENAME: ClassVar[str] = "metadata.json"

    kg: kg.Graph
    semantic: semantic.Graph
    citations: citations.Graph
    encoder_model: str

    @classmethod
    def build(
        cls,
        encoder: emb.Encoder,
        annotated: Sequence[gpt.PaperAnnotated],
        peerread_papers: Sequence[s2.PaperWithS2Refs],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
        *,
        progress: bool = False,
    ) -> None:
        """Build and save all SciMON graphs separately to minimize memory usage.

        Each graph is built, saved to disk, and then cleared from memory before
        building the next one.

        Args:
            encoder: Encoder to use for text embeddings.
            annotated: Annotated papers for KG and semantic graphs.
            peerread_papers: PeerRead papers with S2 references for citation graph.
            output_dir: Directory where to save the graph files.
            metadata: Optional metadata to save with the graph.
            progress: Whether to show progress bars.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building KG graph: %d annotations", len(annotated))
        kg_graph = kg.Graph.from_terms(
            encoder, (x.terms for x in annotated), progress=progress
        )
        kg_data = kg_graph.to_data()
        save_data(output_dir / cls.KG_FILENAME, kg_data)
        # Clear memory
        del kg_graph
        del kg_data
        gc.collect()
        gc.collect()

        logger.info("Building Semantic graph: %d annotations", len(annotated))
        semantic_graph = semantic.Graph.from_annotated(
            encoder, annotated, progress=progress
        )
        semantic_data = semantic_graph.to_data()
        save_data(output_dir / cls.SEMANTIC_FILENAME, semantic_data)
        # Clear memory
        del semantic_graph
        del semantic_data
        gc.collect()
        gc.collect()

        # Build and save Citations graph
        logger.info("Building Citation graph: %d papers", len(peerread_papers))
        citation_graph = citations.Graph.from_papers(
            encoder, peerread_papers, progress=progress
        )
        save_data(output_dir / cls.CITATIONS_FILENAME, citation_graph)

        metadata_model = MetadataModel(
            encoder_model=encoder.model_name,
            metadata=metadata,
        )
        save_data(output_dir / cls.METADATA_FILENAME, metadata_model)

        logger.info("All graphs saved to %s", output_dir)

    @classmethod
    def load(cls, graph_dir: Path) -> Self:
        """Load graph from a directory containing the separate graph files.

        Args:
            graph_dir: Directory containing the graph files.

        Returns:
            A functioning graph, ready to be queried
        """
        # Load metadata first to get encoder model
        metadata_path = graph_dir / cls.METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        metadata = load_data_single(metadata_path, MetadataModel)

        encoder = emb.Encoder(metadata.encoder_model)

        kg_path = graph_dir / cls.KG_FILENAME
        if not kg_path.exists():
            raise FileNotFoundError(f"KG graph file not found at {kg_path}")
        kg_data = load_data_single(kg_path, kg.GraphData)
        kg_graph = kg_data.to_graph(encoder)
        del kg_data

        semantic_path = graph_dir / cls.SEMANTIC_FILENAME
        if not semantic_path.exists():
            raise FileNotFoundError(f"Semantic graph file not found at {semantic_path}")
        semantic_data = load_data_single(semantic_path, semantic.GraphData)
        semantic_graph = semantic_data.to_graph(encoder)
        del semantic_data

        citations_path = graph_dir / cls.CITATIONS_FILENAME
        if not citations_path.exists():
            raise FileNotFoundError(
                f"Citations graph file not found at {citations_path}"
            )
        citation_graph = load_data_single(citations_path, citations.Graph)

        gc.collect()
        gc.collect()

        return cls(
            kg=kg_graph,
            semantic=semantic_graph,
            citations=citation_graph,
            encoder_model=metadata.encoder_model,
        )

    def query_all(
        self,
        ann: gpt.PeerReadAnnotated,
        use_kg: bool = False,
        k: int = CITATION_DEFAULT_K,
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


class QueryResult(Immutable):
    """Query results across graphs, delimited by where they came from."""

    citations: Sequence[str]
    kg: Sequence[str]
    semantic: Sequence[str]


class AnnotatedGraphResult(Record):
    """Annotated PeerRead paper and graph terms queried from it."""

    ann: gpt.PeerReadAnnotated
    result: QueryResult

    @property
    def id(self) -> str:
        """Identify the graph result as the underlying annotated paper's ID."""
        return self.ann.id
