"""Plot representation of a graph with different node types."""

# pyright: basic
import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def main(output_path: Path | None) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()

    nodes = {
        1: "Weak Reward Model Transforms Generative Models into Robust Causal Event Extraction Systems",
        2: "Align Causal Event Extraction Model to human preference using RL",
        3: "Evaluating causal event extraction is not straightforward",
        4: "Train evaluator to align with human preferences",
        5: "Use RL to align extractor with human preferences",
        6: "Primary Area: Natural Language Processing and Machine Learning",
        7: "Keyword: Causal Event Extraction",
        8: "Keyword: Reinforcement Learning",
        9: "Keyword: Reward Modeling",
        10: "Extractor RL PPO training",
        11: "Reward model finetuning",
        12: "Weak supervision for initial model training",
        13: "Evaluation of human-AI agreement on extractions",
        14: "Performance comparison with varying data sizes",
        15: "Analysis of extraction alignment with human preferences",
    }

    edges = [
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 11),
        (3, 13),
        (4, 11),
        (4, 13),
        (5, 10),
        (5, 15),
        (10, 15),
        (11, 13),
        (11, 14),
        (12, 14),
    ]

    type_to_nodes = {
        "Title": [1],
        "TLDR": [2],
        "Claim": [3, 4, 5],
        "Primary Area": [6],
        "Keyword": [7, 8, 9],
        "Method": [10, 11, 12],
        "Experiment Design": [13, 14, 15],
    }

    color_map = {
        "Title": "lightblue",
        "TLDR": "lightgreen",
        "Claim": "lightcoral",
        "Primary Area": "lightyellow",
        "Keyword": "lightsalmon",
        "Method": "lightpink",
        "Experiment Design": "lightgray",
    }

    node_details = {
        3: "Existing metrics may not capture all aspects of causal event extraction quality",
        4: "Human feedback is crucial for aligning the evaluator with human judgment",
        5: "Reinforcement learning can help optimize the extractor for human preferences",
        10: "Implements PPO algorithm to train the extractor using the reward model",
        11: "Iteratively improves the reward model based on human feedback",
        12: "Utilizes large amounts of unlabeled data for initial model training",
        13: "Measures how well the model's extractions align with human judgments",
        14: "Assesses model performance across different training data sizes",
        15: "Evaluates the degree of alignment between extracted events and human expectations",
    }

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

    pos = {
        # Title
        1: (0, 6),
        # TLDR
        2: (0, 5),
        # Claims
        3: (-1.5, 4),
        4: (0, 4),
        5: (1.5, 4),
        # Primary Area
        6: (-2, 5),
        # Keywords
        7: (2, 5.5),
        8: (2, 5),
        9: (2, 4.5),
        # Methods
        10: (-1, 3),
        11: (0, 3),
        12: (1, 3),
        # Experiment Design
        13: (-1, 2),
        14: (0, 2),
        15: (1, 2),
    }

    plt.figure(figsize=(24, 13))

    nx.draw(
        graph,
        pos,
        node_size=3000,
        node_color=node_colors,
        font_size=9,
        font_weight="bold",
        arrows=True,
    )

    labels = {node: label for node, label in nodes.items()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)

    for node, (x, y) in pos.items():
        node_type = None
        for type_name, nodes_list in type_to_nodes.items():
            if node in nodes_list:
                node_type = type_name
                break

        if not node_type:
            continue

        plt.text(
            x,
            y - 0.15,
            node_type,
            fontsize=8,
            ha="center",
            color="black",
            style="italic",
        )
        if (
            node_type in ["Claim", "Method", "Experiment Design"]
            and node in node_details
        ):
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
