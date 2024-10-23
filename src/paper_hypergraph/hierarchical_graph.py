"""Represent a hierarchical graph with directed nodes and edges, each with a string type.

Create, validate, load and save hierarchical graphs, and visualise them as PNG files
or in the GUI.
"""

# pyright: basic
from __future__ import annotations

import argparse
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle


class GraphError(Exception):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    type: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str


class DiGraph:
    """Directed graph with nodes and edges. Nodes and edges have string types."""

    _nxgraph: nx.DiGraph[str]

    def __init__(self, nxgraph: nx.DiGraph[str]) -> None:
        self._nxgraph = nxgraph

    @classmethod
    def from_elements(cls, *, nodes: Iterable[Node], edges: Iterable[Edge]) -> Self:
        """Create a new graph from nodes and edges.

        Nodes are added first, and it's assumed that the edges will connect existing
        nodes.
        """
        nxgraph = nx.DiGraph()

        for node in nodes:
            nxgraph.add_node(node.name, type=node.type)

        for edge in edges:
            nxgraph.add_edge(edge.source, edge.target)

        return cls(nxgraph)

    def visualise_hierarchy(
        self,
        img_path: Path,
        display_gui: bool = True,
        description: str | None = None,
    ) -> None:
        """Visualise a hierarchical directed acyclical graph with matplotlib.

        Saves the visualisation to a file. Optionally, can show the plot in the GUI.
        Note: plotting to GUI suspends the calling thread until the plot is closed.

        Args:
            img_path: Path to save the image of the visualisation.
            display_gui: If True, display the plot in the GUI.
            description: If present, add the description to the plot title.

        Raises:
            GraphError: If the graph doesn't have any root nodes (nodes with degree 0),
                has multiple root nodes, or a cycle.
        """
        nxgraph = self._nxgraph

        # Identify root nodes (nodes with in-degree 0)
        in_degrees = cast(nx.classes.reportviews.DiDegreeView, nxgraph.in_degree())
        roots = [node for node, in_degree in in_degrees if in_degree == 0]

        if not roots:
            raise GraphError(
                "The graph doesn't have any root nodes (nodes with in-degree 0)"
            )
        if len(roots) > 1:
            raise GraphError(
                f"The graph has multiple root nodes. It should have only one."
                f" Found {len(roots)}."
            )
        if not nx.is_directed_acyclic_graph(nxgraph):
            raise GraphError(
                "The graph has a cycle. It should be a directed acyclic graph."
            )

        def node_depth(node: str) -> int:
            if nxgraph.in_degree(node) == 0:
                return 0
            return 1 + max(node_depth(parent) for parent in nxgraph.predecessors(node))

        depths = {node: node_depth(node) for node in nxgraph.nodes()}
        max_depth = max(depths.values())

        # Create Hierarchical position mapping
        pos: dict[str, tuple[float, float]] = {}
        nodes_at_depth: dict[int, list[str]] = {d: [] for d in range(max_depth + 1)}

        for node, depth in depths.items():
            nodes_at_depth[depth].append(node)

        for depth, nodes in nodes_at_depth.items():
            width = len(nodes)
            for i, node in enumerate(nodes):
                pos[node] = ((i - (width - 1) / 2) / max(width - 1, 1), -depth)

        plt.figure(figsize=(20, 12))

        # Draw nodes and labels with wrapped text
        for node, (x, y) in pos.items():
            node_type = nxgraph.nodes[node].get("type", "")

            # Wrap the text to fit in the box
            wrapped_text = textwrap.wrap(node, width=25)

            # Calculate box dimensions
            box_width = 0.15
            # Adjust height based on number of lines
            box_height = 0.05 * (len(wrapped_text) + 1)

            rect = Rectangle(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                fill=True,
                facecolor="lightblue",
                edgecolor="black",
                zorder=1,  # Nodes drawn first
            )
            plt.gca().add_patch(rect)

            # Add wrapped text
            plt.text(
                x,
                y,
                "\n".join(wrapped_text),
                ha="center",
                va="center",
                wrap=True,
                fontsize=8,
                zorder=3,  # Labels drawn above nodes and edges
            )

            # Add node type
            plt.text(
                x,
                y + box_height / 2 + 0.02,
                node_type,
                ha="center",
                va="bottom",
                color="black",
                fontsize=8,
                zorder=3,  # Labels drawn above nodes and edges
            )

        # Draw edges with arrows
        edge_collection = nx.draw_networkx_edges(
            nxgraph,
            pos,
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
        )

        # Set zorder for edges
        if edge_collection:
            if isinstance(edge_collection, list):
                for col in edge_collection:
                    col.set_zorder(2)  # Edges drawn above nodes but below labels
            else:
                edge_collection.set_zorder(2)

        # Draw edge labels
        edge_labels = {
            edge: nxgraph.edges[edge].get("type", "")
            for edge in nxgraph.edges()
            if nxgraph.edges[edge].get("type")
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                nxgraph,
                pos,
                edge_labels=edge_labels,
                label_pos=0.5,
                font_size=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
                verticalalignment="center",
                horizontalalignment="center",
                rotate=False,
            )

        title = "Paper Hierarchical Graph"
        if description:
            title += f"\n{description}"
        plt.title(title)

        plt.axis("off")
        plt.tight_layout()

        plt.savefig(img_path)
        if display_gui:
            plt.show()

    def validate_hierarchy(self) -> str | None:
        """Validate that the graph follows the hirarchical rules.

        Rules:
        1. The graph must have a single root node (in-degree 0).
        2. The graph must be a directed acyclic graph (no cycles).
        3. Each concept node must connect to at least one supporting sentence
           (out-degree > 0).

        Args:
            graph: The graph to validate.

        Returns:
            None if the graph is valid, otherwise a message explaining the violated rule.
        """
        nxgraph = self._nxgraph
        roots = [node for node in nxgraph.nodes if nxgraph.in_degree(node) == 0]

        # 1. Single root node
        if len(roots) != 1:
            return f"The graph must have a single root node. Found {len(roots)}."

        # 2. Must be a DAG
        if not nx.is_directed_acyclic_graph(nxgraph):
            return "The graph has a cycle. It should be a directed acyclic graph."

        concepts = [
            node
            for node, data in nxgraph.nodes(data=True)
            if data.get("type") == "concept"
        ]
        out_degrees = cast(
            nx.classes.reportviews.OutDegreeView, nxgraph.out_degree(concepts)
        )
        concepts_unconnected = sum(
            out_degree == 0 for _, out_degree in out_degrees if out_degree == 0
        )
        # 3. Each concept must connect to at least one supporting sentence
        if concepts_unconnected > 0:
            return (
                "Each concept must connect to at least one supporting sentence."
                f" Found {concepts_unconnected} that don't."
            )

        return None

    def save(self, path: Path) -> None:
        """Save a graph to a GraphML file."""
        if path.suffix != ".graphml":
            path = path.with_suffix(".graphml")
        nx.write_graphml(self._nxgraph, path)

    def graphml(self) -> str:
        """Save the graph as a GraphML string."""
        return "\n".join(nx.generate_graphml(self._nxgraph))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a graph from a GraphML file."""
        return cls(nx.read_graphml(path, node_type=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a hierarchical graph")
    parser.add_argument("graph_file", type=Path, help="Path to the graph file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save the image of the visualisation",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Don't display the visualisation in the GUI",
    )
    args = parser.parse_args()
    graph = DiGraph.load(args.graph_file)
    graph.visualise_hierarchy(display_gui=args.show, img_path=args.output)


if __name__ == "__main__":
    main()
