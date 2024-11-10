"""Demonstrate that a cycle is possible in a graph with the following rules.

- It's a directed graph.
- If A->B exists, B->A cannot exist.
- There are three types of nodes: title, concept, sentence.
- Title can only have outgoing edges to concepts and no incoming edges.
- Concepts can have outgoing edges to other concepts and sentences, but not to titles.
- Sentences can have outgoing edges to other sentences, but not to titles or concepts.

The script creates a graph with the minimum structure that could potentially have a
cycle. It finds a cycle among the sentences where we have S1 -> S2 -> S3 -> S1. Something
similar is possible with concepts.
"""

# pyright: basic

from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx


class NodeType(Enum):
    TITLE = 1
    CONCEPT = 2
    SENTENCE = 3


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeType] = {}
        self.edges: defaultdict[str, set[str]] = defaultdict(set)

    def title_node(self) -> str | None:
        for node, node_type in self.nodes.items():
            if node_type is NodeType.TITLE:
                return node
        return None

    def add_node(self, node: str, node_type: NodeType) -> None:
        if node_type is NodeType.TITLE and self.title_node() is not None:
            raise ValueError("Only one title node is allowed")

        self.nodes[node] = node_type
        self.edges[node] = set()

    def add_edge(self, from_node: str, to_node: str) -> None:
        if self.is_valid_edge(from_node, to_node):
            self.edges[from_node].add(to_node)
        else:
            raise ValueError(f"Invalid edge: {from_node} -> {to_node}")

    def is_valid_edge(self, from_node: str, to_node: str) -> bool:
        type_from = self.nodes[from_node]
        type_to = self.nodes[to_node]

        match (type_from, type_to):
            case (NodeType.TITLE, NodeType.CONCEPT):
                return True
            case (NodeType.CONCEPT, NodeType.CONCEPT | NodeType.SENTENCE):
                return True
            case (NodeType.SENTENCE, NodeType.SENTENCE):
                return True
            case _:
                return False

    def find_cycle(self) -> list[str] | None:
        visited: set[str] = set()
        recursion_stack: list[str] = []
        cycle: list[str] = []

        def _dfs(node: str) -> bool:
            """If DFS finds a node already visited, we have a cycle."""
            visited.add(node)
            recursion_stack.append(node)
            cycle.append(node)

            for neighbour in self.edges[node]:
                if neighbour not in visited and _dfs(neighbour):
                    return True
                elif neighbour in recursion_stack:
                    cycle.append(neighbour)
                    return True

            recursion_stack.pop()
            cycle.pop()
            return False

        for node in self.nodes:
            if node not in visited and _dfs(node):
                return cycle

        return None


def visualise_graph_cycle(graph: Graph, cycle: list[str] | None = None) -> None:
    """Generate a visualization of the graph. Highlights a cycle, if provided.

    Outputs a PNG file "graph.png" in the current directory.
    """
    nxg = nx.DiGraph()

    # Add nodes
    for node, node_type in graph.nodes.items():
        color = {
            NodeType.TITLE: "lightblue",
            NodeType.CONCEPT: "lightgreen",
            NodeType.SENTENCE: "lightyellow",
        }[node_type]
        nxg.add_node(node, color=color)

    # Add edges
    for from_node, to_nodes in graph.edges.items():
        for to_node in to_nodes:
            color = (
                "red" if cycle and from_node in cycle and to_node in cycle else "black"
            )
            nxg.add_edge(from_node, to_node, color=color)

    # Create a layout for the graph
    pos = nx.spring_layout(nxg)

    # Draw the graph
    plt.figure(figsize=(24, 16))
    nx.draw(
        nxg,
        pos,
        with_labels=True,
        node_color=[nxg.nodes[n]["color"] for n in nxg.nodes()],
        edge_color=[nxg[u][v]["color"] for u, v in nxg.edges()],
        node_size=3000,
        font_size=10,
        font_weight="bold",
        arrows=True,
    )

    # Save the graph
    plt.savefig("graph.png")
    plt.close()


def main() -> None:
    # Create a graph with the minimum structure that could potentially have a cycle
    graph = Graph()

    # Add nodes
    graph.add_node("T1", NodeType.TITLE)
    graph.add_node("C1", NodeType.CONCEPT)
    graph.add_node("C2", NodeType.CONCEPT)
    graph.add_node("S1", NodeType.SENTENCE)
    graph.add_node("S2", NodeType.SENTENCE)
    graph.add_node("S3", NodeType.SENTENCE)

    # Add edges
    graph.add_edge("T1", "C1")
    graph.add_edge("T1", "C2")  # Ensure C2 is also connected to the title
    graph.add_edge("C1", "C2")
    graph.add_edge("C2", "S1")
    graph.add_edge("S1", "S2")
    graph.add_edge("S2", "S3")
    graph.add_edge("S3", "S1")  # This creates a valid cycle

    # Check if the graph has a cycle
    cycle = graph.find_cycle()

    # Run the check and visualize the graph
    print(f"Is it possible to construct a graph with a cycle? {bool(cycle)}")
    if cycle:
        print(f"Cycle found: {' -> '.join(cycle)}")

    # Generate and save the visualization
    visualise_graph_cycle(graph, cycle)
    print("Graph visualization saved as 'graph.png'")


if __name__ == "__main__":
    main()
