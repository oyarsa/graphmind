import argparse
import textwrap
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle


def visualise_hierarchy(
    g: nx.DiGraph, show: bool = True, img_path: Path | None = None
) -> None:
    """Visualise a hierarchical graph with matplotlib.

    Args:
        g: The graph to visualise.
        show: Whether to display the plot in the GUI.
        img_path: Path to save the image of the visualisation.
    """

    # Identify root nodes (nodes with in-degree 0)
    in_degrees = cast(nx.classes.reportviews.DiDegreeView, g.in_degree())
    roots = [node for node, in_degree in in_degrees if in_degree == 0]

    if not roots:
        raise ValueError(
            "The graph doesn't have any root nodes (nodes with in-degree 0)"
        )

    # Compute the depth of each node
    def node_depth(node: str) -> int:
        if g.in_degree(node) == 0:
            return 0
        return 1 + max(node_depth(parent) for parent in g.predecessors(node))

    depths: dict[str, int] = {node: node_depth(node) for node in g.nodes()}
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
        node_type = g.nodes[node].get("type", "")

        # Wrap the text to fit in the box
        wrapped_text = textwrap.wrap(node, width=20)

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
            color="red",
            fontsize=8,
            zorder=3,  # Labels drawn above nodes and edges
        )

    # Draw edges with arrows
    edge_collection = nx.draw_networkx_edges(
        g,
        pos,
        edge_color="gray",
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

    plt.title("Paper Hierarchical Graph")
    plt.axis("off")
    plt.tight_layout()

    if img_path:
        plt.savefig(img_path)
    if show:
        plt.show()


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    """Save a graph to a GraphML file."""
    if path.suffix != ".graphml":
        path = path.with_suffix(".graphml")
    nx.write_graphml(graph, path)


def load_graph(path: Path) -> nx.DiGraph:
    """Load a graph from a GraphML file."""
    return nx.read_graphml(path, node_type=str)


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
    graph = load_graph(args.graph_file)
    visualise_hierarchy(graph, show=args.show, img_path=args.output)


if __name__ == "__main__":
    main()
