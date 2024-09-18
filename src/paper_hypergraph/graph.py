import textwrap
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle


def visualise_hierarchy(g: nx.DiGraph) -> None:
    """Visualise a hierarchical graph with matplotlib."""

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

    # Increase vertical spacing between levels
    vertical_spacing = 1.5
    for depth, nodes in nodes_at_depth.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (
                (i - (width - 1) / 2) / max(width - 1, 1),
                -depth * vertical_spacing,
            )

    # Increase figure size and set margins
    plt.figure(figsize=(24, 18))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Draw edges with arrows and curved paths
    for edge in g.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        plt.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                connectionstyle="arc3,rad=0.1",
                alpha=0.6,
                linewidth=1.5,
            ),
        )

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
            zorder=10,  # Ensure nodes are drawn on top of edges
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
            zorder=11,  # Ensure text is drawn on top of nodes
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
            zorder=11,  # Ensure text is drawn on top of nodes
        )

    plt.title("Paper Hierarchical Graph")
    plt.axis("off")
    plt.show()
