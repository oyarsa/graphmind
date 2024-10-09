"""Plot representation of a graph with different node types."""

# pyright: basic
import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def main(output_path: Path | None) -> None:
    # Creating a directed graph to represent the flowchart
    graph = nx.DiGraph()

    # Adding nodes based on the structure in the image
    nodes = {
        1: "Weak Reward Model Transforms Generative Models into Robust Causal Event"
        " Extraction Systems",
        2: "Align Causal Event Extraction Model to human preference using RL",
        3: "Evaluating causal event extraction is not straightforward",
        4: "Train evaluator to align with human preferences",
        5: "Use RL to align extractor with human preferences",
        6: "Train evaluator with human evaluation data",
        7: "Use PPO to align extractor with human preferences",
        8: "Weak-to-strong framework to train reward model",
        9: "Extractor RL PPO training",
        10: "Reward model finetuning",
        11: "Weak supervision",
        12: "High evaluation agreement",
        13: "High performance with less data",
        14: "Highly aligned extraction output",
    }

    # Adding edges for the flow
    edges = [
        (1, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (4, 6),
        (4, 7),
        (5, 7),
        (6, 8),
        (6, 10),
        (7, 9),
        (8, 10),
        (8, 11),
        (9, 14),
        (10, 12),
        (10, 14),
        (11, 13),
    ]

    # Defining node types as per the notes (T), (B), etc.
    type_to_nodes = {
        "Title": [1],
        "Problem": [2],
        "Claim": [3, 4, 5],
        "Contribution": [6, 7, 8],
        "Method": [9, 10, 11],
        "Result": [12, 13, 14],
    }

    # Define colors for different node types
    color_map = {
        "Title": "lightblue",
        "Problem": "lightgreen",
        "Claim": "lightcoral",
        "Contribution": "lightyellow",
        "Method": "lightpink",
        "Result": "lightgray",
    }

    # Assign two sentences to each node
    node_details: dict[int, str] = {}
    for i, node in enumerate(nodes):
        if i == 0:  # Skip the Title node
            continue
        node_details[node] = f"Placeholder details #{i}"

    # Creating a node-to-type mapping from type-to-nodes
    node_colors: list[str] = []
    node_types: list[str] = []
    for node in nodes:
        for node_type, node_list in type_to_nodes.items():
            if node in node_list:
                node_colors.append(color_map[node_type])
                node_types.append(node_type)

    for node, label in nodes.items():
        graph.add_node(node, label=label)

    graph.add_edges_from(edges)

    # Manually positioning nodes for hierarchical layout
    pos = {
        # Title
        1: (0, 5),
        # Problem
        2: (0, 4),
        # Claims
        3: (-1.5, 3),
        4: (0, 3),
        5: (1.5, 3),
        # Contributions
        6: (-2, 2),
        7: (0, 2),
        8: (2, 2),
        # Methods
        9: (-1, 1),
        10: (0, 1),
        11: (1, 1),
        # Results
        12: (-1, 0),
        13: (0, 0),
        14: (1, 0),
    }

    plt.figure(figsize=(24, 13))

    # Draw nodes (without showing node numbers) and arrows
    nx.draw(
        graph,
        pos,
        node_size=3000,
        node_color=node_colors,
        font_size=9,
        font_weight="bold",
        arrows=True,
    )

    # Adding only the labels (no numbers)
    labels = {node: label for node, label in nodes.items()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)

    # Adding small text on each node showing the type and detail
    for node, (x, y) in pos.items():
        node_type = None
        for type_name, nodes_list in type_to_nodes.items():
            if node in nodes_list:
                node_type = type_name
                break
        if node_type:
            plt.text(
                x,
                y - 0.15,
                node_type,
                fontsize=8,
                ha="center",
                color="black",
                style="italic",
            )
            # Add detail property for all nodes except Title
            if node_type != "Title":
                detail_text = node_details[node]
                wrapped_text = textwrap.fill(detail_text, width=40)
                plt.text(
                    x,
                    y - 0.25,
                    wrapped_text,
                    fontsize=6,
                    ha="center",
                    va="top",
                    color="black",
                    wrap=True,
                )

    plt.title("Paper Graph", fontsize=14)

    if output_path:
        plt.savefig(output_path)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save the figure."
    )
    args = parser.parse_args()
    main(args.output)
