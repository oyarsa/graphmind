#!/usr/bin/env python3
"""Plot representation of a graph with different node types."""

from pathlib import Path

from paper.hierarchical_graph import DiGraph, Edge, Node
from paper.util import HelpOnErrorArgumentParser


def main(output: Path | None) -> None:
    node_names = {
        "title": {
            "main": "Weak Reward Model Transforms Generative Models into Robust Causal"
            " Event Extraction Systems",
        },
        "tldr": {
            "main": "Align Causal Event Extraction Model to human preference using RL",
        },
        "area": {
            "main": "Primary Area: Natural Language Processing and Machine Learning",
        },
        "keywords": {
            "causal": "Keyword: Causal Event Extraction",
            "rl": "Keyword: Reinforcement Learning",
            "reward": "Keyword: Reward Modeling",
        },
        "claims": {
            "eval": "Evaluating causal event extraction is not straightforward",
            "train": "Train evaluator to align with human preferences",
            "rl": "Use RL to align extractor with human preferences",
        },
        "methods": {
            "ppo": "Extractor RL PPO training",
            "reward": "Reward model finetuning",
            "weak": "Weak supervision for initial model training",
        },
        "experiments": {
            "agree": "Evaluation of human-AI agreement on extractions",
            "perf": "Performance comparison with varying data sizes",
            "align": "Analysis of extraction alignment with human preferences",
        },
    }

    node_details = {
        node_names["claims"][
            "eval"
        ]: "Existing metrics may not capture all aspects of causal event extraction quality",
        node_names["claims"][
            "train"
        ]: "Human feedback is crucial for aligning the evaluator with human judgment",
        node_names["claims"][
            "rl"
        ]: "Reinforcement learning can help optimize the extractor for human preferences",
        node_names["methods"][
            "ppo"
        ]: "Implements PPO algorithm to train the extractor using the reward model",
        node_names["methods"][
            "reward"
        ]: "Iteratively improves the reward model based on human feedback",
        node_names["methods"][
            "weak"
        ]: "Utilizes large amounts of unlabeled data for initial model training",
        node_names["experiments"][
            "agree"
        ]: "Measures how well the model's extractions align with human judgments",
        node_names["experiments"][
            "perf"
        ]: "Assesses model performance across different training data sizes",
        node_names["experiments"][
            "align"
        ]: "Evaluates the degree of alignment between extracted events and human expectations",
    }

    nodes = [
        Node(name=node_names["title"]["main"], type="title", detail=""),
        Node(name=node_names["tldr"]["main"], type="tldr", detail=""),
        Node(name=node_names["area"]["main"], type="primary_area", detail=""),
        *[
            Node(name=name, type="keyword", detail="")
            for name in node_names["keywords"].values()
        ],
        *[
            Node(name=name, type="claim", detail=node_details[name])
            for name in node_names["claims"].values()
        ],
        *[
            Node(name=name, type="method", detail=node_details[name])
            for name in node_names["methods"].values()
        ],
        *[
            Node(name=name, type="experiment", detail=node_details[name])
            for name in node_names["experiments"].values()
        ],
    ]

    edges = [
        # Title connections
        *[Edge(node_names["title"]["main"], node_names["tldr"]["main"])],
        *[Edge(node_names["title"]["main"], node_names["area"]["main"])],
        *[
            Edge(node_names["title"]["main"], keyword)
            for keyword in node_names["keywords"].values()
        ],
        # TLDR to Claims
        *[
            Edge(node_names["tldr"]["main"], claim)
            for claim in node_names["claims"].values()
        ],
        # Claims to Methods
        Edge(node_names["claims"]["eval"], node_names["methods"]["reward"]),
        Edge(node_names["claims"]["train"], node_names["methods"]["reward"]),
        Edge(node_names["claims"]["rl"], node_names["methods"]["ppo"]),
        # Methods to Experiments
        Edge(node_names["methods"]["ppo"], node_names["experiments"]["align"]),
        Edge(node_names["methods"]["reward"], node_names["experiments"]["agree"]),
        Edge(node_names["methods"]["reward"], node_names["experiments"]["perf"]),
        Edge(node_names["methods"]["weak"], node_names["experiments"]["perf"]),
    ]

    graph = DiGraph.from_elements(nodes=nodes, edges=edges)
    graph.visualise_hierarchy(output)


if __name__ == "__main__":
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument("output", type=Path, help="Path to save the figure ")
    args = parser.parse_args()
    main(args.output)
