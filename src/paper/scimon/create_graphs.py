from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from paper.scimon.model import Paper


@dataclass(frozen=True, kw_only=True)
class Node:
    """A node in the graph with its attributes."""

    id: str
    attributes: Mapping[str, str]


@dataclass(frozen=True, kw_only=True)
class Edge:
    """An edge in the graph with its weight and attributes."""

    source: str
    target: str
    weight: float
    attributes: Mapping[str, str]


class Graph:
    """A graph representation using adjacency lists."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, set[Edge]] = {}

    def add_node(self, node_id: str, **attributes: str) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            **attributes: Additional attributes for the node.

        Raises:
            ValueError: If node_id already exists.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        self.nodes[node_id] = Node(id=node_id, attributes=attributes)
        self.edges[node_id] = set()

    def add_edge(
        self, source: str, target: str, weight: float = 1.0, **attributes: str
    ) -> None:
        """Add a weighted edge between two nodes.

        Args:
            source: ID of the source node.
            target: ID of the target node.
            weight: Edge weight.
            **attributes: Additional attributes for the edge.

        Raises:
            ValueError: If either source or target node doesn't exist.
        """
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge")

        edge = Edge(source=source, target=target, weight=weight, attributes=attributes)
        self.edges[source].add(edge)


def compute_embedding_similarity(
    emb1: npt.NDArray[np.float64], emb2: npt.NDArray[np.float64]
) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding vector.
        emb2: Second embedding vector.

    Returns:
        Cosine similarity score between the embeddings.
    """
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def build_semantic_graph(
    papers: Iterable[Paper], encoder: SentenceTransformer
) -> Graph:
    """Build semantic similarity graph from papers.

    Args:
        papers: Iterable of papers with their terms and background.
        encoder: SentenceTransformer model for computing embeddings.

    Returns:
        Graph with nodes representing task-method pairs and edges weighted by semantic
        similarity.
    """
    graph = Graph()
    node_embeddings: dict[str, npt.NDArray[np.float64]] = {}

    # Create nodes and compute embeddings
    for paper in papers:
        for relation in paper.terms.relations:
            base_input = (
                f"'{relation.head}' -> '{relation.tail}' | context: {paper.background}"
            )
            node_id = f"paper_{paper.id}_{relation.head}_{relation.tail}"

            graph.add_node(
                node_id,
                paper_id=paper.id,
                term1=relation.head,
                term2=relation.tail,
                context=paper.background,
                base_input=base_input,
            )

            node_embeddings[node_id] = encoder.encode(base_input).numpy()  # type: ignore

    # Create edges based on semantic similarity
    for node1_id in graph.nodes:
        for node2_id in graph.nodes:
            if node1_id != node2_id:
                similarity = compute_embedding_similarity(
                    node_embeddings[node1_id],
                    node_embeddings[node2_id],
                )
                graph.add_edge(node1_id, node2_id, weight=similarity)

    return graph


def build_knowledge_graph(papers: Iterable[Paper]) -> Graph:
    """Build knowledge graph connecting terms across papers.

    The entities in the scientific terms (both the terms themselves and relations heads
    and tails) are casefolded. If they are present in the abstract (also casefolded),
    they are added (in original form) to the graph.

    Args:
        papers: Iterable of papers with their terms and context.

    Returns:
        Graph with nodes representing terms and edges representing relations.
    """
    graph = Graph()

    for paper in papers:
        abstract_folded = paper.abstract.casefold()

        entities_types = [
            (paper.terms.tasks, "task"),
            (paper.terms.methods, "method"),
            (paper.terms.metrics, "metric"),
            (paper.terms.resources, "material"),
        ]
        for entities, type in entities_types:
            for entity in entities:
                entity_f = entity.casefold()
                if entity_f in abstract_folded:
                    graph.add_node(entity, type=type)

        for relation in paper.terms.relations:
            head_valid = relation.head.casefold() in abstract_folded
            tail_valid = relation.tail.casefold() in abstract_folded
            if head_valid and tail_valid:
                graph.add_edge(relation.head, relation.tail, paper_id=paper.id)

    return graph
