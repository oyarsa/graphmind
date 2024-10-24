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
from typing import Self

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle


class GraphError(Exception):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    type: str
    detail: str = ""


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
            nxgraph.add_node(node.name, type=node.type, detail=node.detail)

        for edge in edges:
            nxgraph.add_edge(edge.source, edge.target)

        return cls(nxgraph)

    def visualise_hierarchy(
        self,
        img_path: Path | None = None,
        display_gui: bool = True,
        description: str | None = None,
    ) -> None:
        """Visualize a paper graph following the hierarchical structure rules."""
        if img_path is None and not display_gui:
            raise ValueError(
                "Either `img_path` must be provided or `display_gui` must be True"
            )

        nxgraph = self._nxgraph
        plt.figure(figsize=(20, 12))

        node_types = nx.get_node_attributes(nxgraph, "type")
        node_details = nx.get_node_attributes(nxgraph, "detail")

        # Level 1: Validate and find Title node
        title_nodes = [node for node, type_ in node_types.items() if type_ == "Title"]
        if len(title_nodes) != 1:
            raise GraphError(
                f"Graph must have exactly one Title node. Found {len(title_nodes)}"
            )
        title_node = title_nodes[0]

        # Group nodes by levels
        levels: dict[int, list[tuple[str, str]]] = {
            1: [(title_node, "Title")],
            2: [],  # Primary Area, Keywords, TLDR
            3: [],  # Claims
            4: [],  # Methods
            5: [],  # Experiments
        }

        # Level 2: Primary Area, Keywords, TLDR
        level2_types = {"Primary Area", "Keyword", "TLDR"}
        for node, type_ in node_types.items():
            if type_ in level2_types:
                predecessors = list(nxgraph.predecessors(node))
                if len(predecessors) != 1 or predecessors[0] != title_node:
                    raise GraphError(
                        f"Level 2 node {node} must have exactly one incoming edge"
                        " from Title"
                    )
                levels[2].append((node, type_))

        # Find TLDR node for next level
        tldr_nodes = [
            (node, type_) for node, type_ in node_types.items() if type_ == "TLDR"
        ]
        if len(tldr_nodes) != 1:
            raise GraphError(
                f"Graph must have exactly one TLDR node. Found {len(tldr_nodes)}"
            )
        tldr_node = tldr_nodes[0][0]

        # Level 3: Claims
        claim_nodes = [
            (node, type_) for node, type_ in node_types.items() if type_ == "Claim"
        ]
        for node, type_ in claim_nodes:
            predecessors = list(nxgraph.predecessors(node))
            if len(predecessors) != 1 or predecessors[0] != tldr_node:
                raise GraphError(
                    f"Claim node {node} must have exactly one incoming edge from TLDR. "
                    f"Found predecessors: {predecessors}"
                )
            levels[3].append((node, type_))

        # Level 4: Methods
        method_nodes = [
            (node, type_) for node, type_ in node_types.items() if type_ == "Method"
        ]
        for node, type_ in method_nodes:
            predecessors = list(nxgraph.predecessors(node))
            if not all(node_types[pred] == "Claim" for pred in predecessors):
                raise GraphError(
                    f"Method node {node} must only have incoming edges from Claims"
                )
            levels[4].append((node, type_))

        # Level 5: Experiments
        experiment_nodes = [
            (node, type_)
            for node, type_ in node_types.items()
            if type_ == "Experiment Design"
        ]
        for node, type_ in experiment_nodes:
            predecessors = list(nxgraph.predecessors(node))
            if not all(node_types[pred] == "Method" for pred in predecessors):
                raise GraphError(
                    f"Experiment node {node} must only have incoming edges from Methods"
                )
            levels[5].append((node, type_))

        # Calculate positions
        pos: dict[str, tuple[float, float]] = {}
        colors: dict[str, str] = {}
        color_map = {
            "Title": "lightblue",
            "TLDR": "lightgreen",
            "Claim": "lightcoral",
            "Primary Area": "lightyellow",
            "Keyword": "lightsalmon",
            "Method": "lightpink",
            "Experiment Design": "lightgray",
        }

        # Position nodes by level
        for level, nodes in levels.items():
            if not nodes:
                continue

            nodes.sort(key=lambda x: x[0])
            y_pos = -(level - 1) * 2.0  # Increased vertical spacing for details
            width = len(nodes)

            for i, (node, type_) in enumerate(nodes):
                x_pos = (i - (width - 1) / 2) * 1.8  # Increased horizontal spacing
                pos[node] = (x_pos, y_pos)
                colors[node] = color_map[type_]

        # Draw nodes
        for node, (x, y) in pos.items():
            # Get node info
            detail = node_details.get(node)

            # Wrap node text
            wrapped_text = textwrap.fill(node, width=20)
            num_lines = len(wrapped_text.split("\n"))

            # Get detail if it exists and wrap it
            detail_lines = 0
            if detail:
                wrapped_detail = textwrap.fill(str(detail), width=30)
                detail_lines = len(wrapped_detail.split("\n"))

            # Adjust box size based on content
            box_width = 1.0
            box_height = 0.3 + (num_lines * 0.15) + (detail_lines * 0.12)

            # Create node box
            rect = Rectangle(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                fill=True,
                facecolor=colors[node],
                edgecolor="black",
                zorder=1,
                alpha=0.7,
            )
            plt.gca().add_patch(rect)

            # Add node type at the top
            plt.text(
                x,
                y + box_height / 2 + 0.1,
                node_types[node],
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                style="italic",
                zorder=3,
            )

            # Add node name
            name_y = y + (detail_lines * 0.06) if detail is not None else y
            plt.text(
                x,
                name_y,
                wrapped_text,
                ha="center",
                va="center",
                fontsize=9,
                wrap=True,
                zorder=3,
            )

            # Add detail if it exists
            if detail is not None:
                plt.text(
                    x,
                    y - (num_lines * 0.06) - 0.1,
                    wrapped_detail,
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="darkslategray",
                    wrap=True,
                    style="italic",
                    zorder=3,
                )

        # Draw edges
        for edge in nxgraph.edges():
            source_pos = pos[edge[0]]
            target_pos = pos[edge[1]]
            rad = 0.2 if abs(source_pos[1] - target_pos[1]) > 1 else 0.1

            nx.draw_networkx_edges(
                nxgraph,
                pos,
                edgelist=[edge],
                arrows=True,
                arrowsize=15,
                arrowstyle="->",
                connectionstyle=f"arc3,rad={rad}",
                edge_color="gray",
                alpha=0.6,
                width=1,
            )

        title = "Paper Hierarchical Graph"
        if description:
            title += f"\n{description}"
        plt.title(title)

        plt.axis("off")
        plt.tight_layout()

        if img_path:
            plt.savefig(img_path)
        if display_gui:
            plt.show()

    def has_cycle(self) -> bool:
        return not nx.is_directed_acyclic_graph(self._nxgraph)

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
