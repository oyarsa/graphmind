"""Remove cycles from a graph by removing the Minimum Weight Feedback Arc Set.

Given a directed graph of weighted edges that might contain cycles, find the subset
of edges that when removed, undo all the cycles in a way that the sum of weights of
these edges is minimised.

It's a NP-hard (decision problem is NP-complete) problem, but since our graphs are
small (~20 nodes), we can solve it through brute force.
"""

from collections import defaultdict
from enum import Enum
from itertools import combinations


class Graph:
    def __init__(self) -> None:
        self.nodes: set[int] = set()
        self.edges: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)

    def add_edge(self, u: int, v: int, weight: int) -> None:
        """Add weighted directed edge u -> v."""
        self.nodes.update((u, v))
        self.edges[u].append((v, weight))

    def remove_edge(self, u: int, v: int) -> None:
        """Remove edge u -> v. Does not remove v even if this was the last edge to it."""
        self.edges[u] = [(w, weight) for w, weight in self.edges[u] if w != v]

    def has_cycle(self) -> bool:
        """Check if the graph contains at least one cycle using DFS.

        Walks the graph using DFS, and if it finds a graph that's currently being
        explored (i.e. recursed over), it means there's a cycle.
        """

        class NodeState(Enum):
            UNVISITED = 0
            VISITING = 1
            VISITED = 2

        node_states = {node: NodeState.UNVISITED for node in self.nodes}

        def dfs(node: int) -> bool:
            node_states[node] = NodeState.VISITING

            for neighbor, _ in self.edges[node]:
                if node_states[neighbor] is NodeState.UNVISITED:
                    if dfs(neighbor):
                        return True
                elif node_states[neighbor] is NodeState.VISITING:
                    # If we reach a node we're currently visiting, it means we've found
                    # a cycle.
                    return True

            node_states[node] = NodeState.VISITED
            return False

        for node in self.nodes:
            if node_states[node] is NodeState.UNVISITED:  # noqa: SIM102
                if dfs(node):
                    return True

        return False

    def total_weight(self) -> int:
        """Sum of the weights for all edges in the graph."""
        return sum(weight for edges in self.edges.values() for _, weight in edges)


def min_weight_feedback_arc_set_naive(graph: Graph) -> Graph:
    """Remove all cycles from the graph in a way that minimises the lost edge weights."""
    arcs_all = [(u, v, weight) for u in graph.edges for v, weight in graph.edges[u]]

    min_weight = int(1e9)
    min_graph = graph

    for set_size in range(len(arcs_all)):
        for arcs_feedback in combinations(arcs_all, set_size):
            candidate = Graph()
            total_weight = 0
            for u, v, weight in arcs_all:
                if (u, v, weight) not in arcs_feedback:
                    candidate.add_edge(u, v, weight)
                else:
                    total_weight += weight

            if not candidate.has_cycle() and total_weight < min_weight:
                min_weight = total_weight
                min_graph = candidate

    return min_graph


def generate_test_cases() -> list[Graph]:
    test_cases: list[Graph] = []

    # Test case 1: Simple cycle
    g1 = Graph()
    g1.add_edge(0, 1, 2)
    g1.add_edge(1, 2, 3)
    g1.add_edge(2, 0, 1)
    test_cases.append(g1)

    # Test case 2: Larger graph with multiple cycles
    g2 = Graph()
    g2.add_edge(0, 1, 5)
    g2.add_edge(1, 2, 3)
    g2.add_edge(2, 3, 2)
    g2.add_edge(3, 0, 4)
    g2.add_edge(1, 3, 1)
    g2.add_edge(3, 1, 2)
    test_cases.append(g2)

    # Test case 3: Directed acyclic graph (DAG)
    g3 = Graph()
    g3.add_edge(0, 1, 1)
    g3.add_edge(1, 2, 2)
    g3.add_edge(2, 3, 3)
    test_cases.append(g3)

    # Test case 4: Graph with self-loop
    g4 = Graph()
    g4.add_edge(0, 1, 2)
    g4.add_edge(1, 1, 1)
    g4.add_edge(1, 2, 3)
    test_cases.append(g4)

    return test_cases


def main() -> None:
    graph_examples = generate_test_cases()

    for i, graph_old in enumerate(graph_examples, 1):
        old_weight = graph_old.total_weight()
        old_num_edges = len(graph_old.edges)
        old_num_nodes = len(graph_old.nodes)

        graph_new = min_weight_feedback_arc_set_naive(graph_old)
        assert old_num_nodes == len(graph_new.nodes)

        new_weight = graph_new.total_weight()
        new_num_edges = len(graph_new.edges)

        print(
            f"{i}/{len(graph_examples)})  nodes={old_num_nodes:x} {old_weight=:2}"
            f" {new_weight=:2} {old_num_edges=} {new_num_edges=}"
        )


if __name__ == "__main__":
    main()
