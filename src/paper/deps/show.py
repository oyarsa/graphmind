"""Build dependency graph from dependency file.

Display the graph and/or save to a file.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, no_type_check

import matplotlib.pyplot as plt
import networkx as nx
import typer
import yaml

from paper.util import read_resource

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


_DEPS_FILE = read_resource("deps", "deps.yaml")


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path | None,
        typer.Argument(help="Dependecy YAML file. Defaults to `paper.deps.deps.yaml`."),
    ] = None,
    show: Annotated[bool, typer.Option(help="Show dependency graph.")] = False,
    save: Annotated[
        Path | None, typer.Option(help="File to save the dependency graph")
    ] = None,
) -> None:
    data = yaml.safe_load(input_file.read_text() if input_file else _DEPS_FILE)

    deps = [
        Dependency(
            source=_file_to_module(entry["source"]),
            target=_file_to_module(entry["target"]),
            detail=entry.get("detail"),
        )
        for entry in data
    ]

    for i, item in enumerate(deps, start=1):
        print(f"{i}. {item}")

    root_nodes = find_root_nodes(deps)
    if root_nodes:
        print("\nRoot nodes (no parents):")
        for node in sorted(root_nodes):
            print(f"- {node}")

    if show or save:
        display_graph(deps, show=show, save_file=save)


@dataclass(frozen=True, kw_only=True)
class Dependency:
    """Relation between a script that generates a file an another that consumes it."""

    source: str
    """Script that generates a file."""
    target: str
    """Script that consumes the generated file."""
    detail: str | None
    """Extra information, such as the subcommand used."""


def find_root_nodes(deps: Sequence[Dependency]) -> set[str]:
    """Find nodes that don't have any parent nodes (root nodes)."""
    all_nodes = {dep.source for dep in deps} | {dep.target for dep in deps}
    has_parent = {dep.target for dep in deps}
    return all_nodes - has_parent


@no_type_check
def display_graph(
    deps: Sequence[Dependency], *, show: bool, save_file: Path | None
) -> None:
    graph: nx.DiGraph[str] = nx.DiGraph()
    edge_labels: dict[tuple[str, str], str] = {}

    for dep in deps:
        graph.add_edge(dep.source, dep.target)
        if dep.detail:
            edge_labels[(dep.source, dep.target)] = dep.detail

    # Calculate levels using longest path from roots
    roots = [n for n, d in graph.in_degree() if d == 0]
    levels: dict[str, int] = {}
    for node in graph.nodes():
        max_dist = 0
        for root in roots:
            try:
                paths = list(nx.all_simple_paths(graph, root, node))
                if paths:
                    max_dist = max(max_dist, len(max(paths, key=len)))
            except nx.NetworkXNoPath:
                continue
        levels[node] = max_dist

    # Position nodes with more spacing
    pos: dict[str, tuple[float, float]] = {}
    level_nodes: dict[int, list[str]] = {}

    # Group nodes by level
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)

    # Position nodes level by level
    max_width = max(len(nodes) for nodes in level_nodes.values())
    for level, nodes in level_nodes.items():
        nodes.sort()  # Sort nodes alphabetically for consistent layout
        for i, node in enumerate(nodes):
            # Center nodes at each level
            x = (i - (len(nodes) - 1) / 2) * 1.5  # Increased horizontal spacing
            y = -level * 2.5  # Increased vertical spacing
            pos[node] = (x, y)

    # Setup larger figure
    plt.figure(figsize=(24, 13))
    plt.clf()

    # Draw edges with less curvature and thicker arrows
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="gray",
        alpha=0.7,
        arrows=True,
        arrowsize=20,
        width=1.5,
        connectionstyle="arc3,rad=0.05",  # Reduced curve
        min_target_margin=25,  # Space for arrows
    )

    # Add edge labels if present
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
            label_pos=0.6,
        )

    # Node colors
    node_colors = [
        "red"
        if graph.in_degree(node) == 0
        else "green"
        if graph.out_degree(node) == 0
        else "lightblue"
        for node in graph.nodes()
    ]

    # Draw larger nodes
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=3000, alpha=0.8, linewidths=2
    )

    # Clearer labels with more padding
    labels = {
        node: "\n".join(node.split(".")[-2:]) if "." in node else node
        for node in graph.nodes()
    }

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
        font_family="sans-serif",
        bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.9, "pad": 6.0},
    )

    plt.title("Data Dependency Graph", pad=20, size=18)
    plt.axis("off")
    plt.tight_layout(pad=2.0)

    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def _file_to_module(file: str) -> str:
    """Convert path to script to its module name."""
    return file.removeprefix("./").removesuffix(".py").replace("/", ".")


if __name__ == "__main__":
    app()
