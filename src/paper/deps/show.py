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
            source=_file_to_pkg(entry["source"]),
            target=_file_to_pkg(entry["target"]),
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
    """Create and display a directed graph.

    Uses custom vertical layout and fixed root positions.
    """
    graph: nx.DiGraph[str] = nx.DiGraph()

    for dep in deps:
        graph.add_edge(dep.source, dep.target)

    # Calculate levels for each node based on longest path from root
    roots = [n for n, d in graph.in_degree() if d == 0]
    levels: dict[str, int] = {}
    for node in graph.nodes():
        # Find the longest path length to any root
        max_dist = 0
        for root in roots:
            try:
                paths = list(nx.all_simple_paths(graph, root, node))
                if paths:
                    max_dist = max(max_dist, len(max(paths, key=len)))
            except nx.NetworkXNoPath:
                continue
        levels[node] = max_dist

    # Create position dictionary
    pos: dict[str, tuple[int, int]] = {}
    level_counts: dict[str, int] = {}  # Count nodes at each level
    level_current: dict[str, int] = {}  # Current count at each level

    # Count nodes per level
    for level in levels.values():
        level_counts[level] = level_counts.get(level, 0) + 1
        level_current[level] = 0

    # Custom positioning for root nodes
    root_positions = {
        "neulab.ReviewAdvisor": -0.5,  # Position on the left
        "ICLR2024.CallForPapers": 0.5,  # Position on the right
    }

    # Position nodes
    sorted_nodes = sorted(levels.items(), key=lambda x: (x[1], x[0]))
    for node, level in sorted_nodes:
        count = level_counts[level]
        current = level_current[level]

        if level == 1 and node in root_positions:
            # Use predetermined position for root nodes
            x = root_positions[node]
        # Calculate x position to center nodes at each level
        elif count > 1:
            x = current - (count - 1) / 2
        else:
            x = 0

        pos[node] = (x, -level)  # Negative level for top-to-bottom layout
        level_current[level] += 1

    # Draw the graph
    plt.figure(figsize=(20, 9))
    plt.clf()

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="gray",
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    # Determine node colors
    node_colors = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:  # Root node
            node_colors.append("red")
        elif graph.out_degree(node) == 0:  # Leaf node
            node_colors.append("green")
        else:  # Internal node
            node_colors.append("lightblue")

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=2500, alpha=0.7, linewidths=1.5
    )

    # Improve label clarity
    labels = {
        node: "\n".join(node.split(".")[-2:]) if "." in node else node
        for node in graph.nodes()
    }

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=12,
        font_weight="bold",
        font_family="sans-serif",
        bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.8, "pad": 3.0},
    )

    plt.title("Dependency Graph", pad=20, size=16)
    plt.margins(x=0.2, y=0.2)
    plt.axis("off")
    plt.tight_layout()

    if save_file:
        plt.savefig(
            save_file,
            bbox_inches="tight",
            metadata={"Creator": "", "Producer": ""},
            facecolor="white",
            edgecolor="none",
        )
    if show:
        plt.show()


def _file_to_pkg(file: str) -> str:
    return file.removeprefix("./").removesuffix(".py").replace("/", ".")


if __name__ == "__main__":
    app()
