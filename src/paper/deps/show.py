"""Build dependency graph from dependency file using Mermaid."""

import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

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


class Method(StrEnum):
    """Method for which to plot the dependencies."""

    PETER = "peter"
    SCIMON = "scimon"

    def data(self) -> str:
        """Contents of the dependency file."""
        return read_resource("deps", f"deps-{self.value}.yaml")


@app.command(help=__doc__)
def main(
    method: Annotated[
        Method,
        typer.Option(
            "--method", "-m", help="What method for which to plot the dependencies"
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="File to save the graph.")
    ],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show extra information.")
    ] = False,
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Show paths to this target node"),
    ] = None,
) -> None:
    """Build dependency graph from dependency file using Mermaid and save to output."""
    data = yaml.safe_load(method.data())

    deps = [
        Dependency(
            source=_file_to_module(entry["source"]),
            target=_file_to_module(entry["target"]),
            detail=entry.get("detail"),
        )
        for entry in data
    ]

    if verbose:
        for i, item in enumerate(deps, start=1):
            print(f"{i}. {item}")

        root_nodes, leaf_nodes = _find_roots_and_leaves(deps)
        print("\nRoot nodes (no parents):")
        for node in sorted(root_nodes):
            print(f"- {node}")

        print("\nLeaf nodes (no children):")
        for node in sorted(leaf_nodes):
            print(f"- {node}")

    if target:
        path = find_paths_to_node(deps, target)
        print(f"\nRequired execution order to reach {target}:")
        print("\n".join(path))

    _save_mermaid(_generate_mermaid(deps), output_file)


@dataclass(frozen=True, kw_only=True)
class Dependency:
    """Relation between a script that generates a file an another that consumes it."""

    source: str
    """Script that generates a file."""
    target: str
    """Script that consumes the generated file."""
    detail: str | None
    """Extra information, such as the subcommand used."""


def find_paths_to_node(deps: Sequence[Dependency], target_node: str) -> list[str]:
    """Find a single path that includes all dependencies to reach the target node."""
    from collections import deque

    # Build forward and reverse adjacency lists
    graph: dict[str, set[str]] = defaultdict(set)
    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for dep in deps:
        graph[dep.source].add(dep.target)
        reverse_graph[dep.target].add(dep.source)

    # Find all required nodes by working backwards from target
    required = {target_node}
    queue = deque([target_node])
    while queue:
        node = queue.popleft()
        for parent in reverse_graph[node]:
            if parent not in required:
                required.add(parent)
                queue.append(parent)

    # Topological sort of required nodes
    in_degree: dict[str, int] = defaultdict(int)
    for node in required:
        for child in graph[node]:
            if child in required:
                in_degree[child] += 1

    queue = deque([node for node in required if in_degree[node] == 0])
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for child in graph[node]:
            if child in required:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    return result


def _file_to_module(file: str) -> str:
    """Convert path to script to its module name."""
    return file.removeprefix("./").removesuffix(".py").replace("/", ".")


def _find_roots_and_leaves(deps: Sequence[Dependency]) -> tuple[set[str], set[str]]:
    """Find nodes that don't have any parents (roots) or children (leaves)."""
    all_nodes = {dep.source for dep in deps} | {dep.target for dep in deps}
    has_parent = {dep.target for dep in deps}
    has_child = {dep.source for dep in deps}

    roots = all_nodes - has_parent
    leaves = all_nodes - has_child
    return roots, leaves


def _generate_mermaid(deps: Sequence[Dependency]) -> str:
    """Generate a Mermaid graph diagram from dependencies."""
    name_to_id: dict[str, int] = {}
    next_id = 0

    lines = ["graph TD"]
    for dep in deps:
        src_id = name_to_id.setdefault(dep.source, (next_id := next_id + 1))
        tgt_id = name_to_id.setdefault(dep.target, (next_id := next_id + 1))

        detail = dep.detail.replace(r"\n", "<br>") if dep.detail else None
        lines.append(
            f'    {src_id}["{dep.source}"]-->|"{detail}"|{tgt_id}["{dep.target}"]'
            if dep.detail
            else f'    {src_id}["{dep.source}"]-->{tgt_id}["{dep.target}"]'
        )

    lines.append("")
    lines.append("    classDef root fill:#FFB6C1")
    lines.append("    classDef leaf fill:#90EE90")

    roots, leaves = _find_roots_and_leaves(deps)
    for class_, nodes in [("root", roots), ("leaf", leaves)]:
        lines.extend(f"    class {name_to_id[node]} {class_}" for node in nodes)

    return "\n".join(lines)


def _save_mermaid(diagram: str, output_file: Path) -> None:
    """Convert a Mermaid diagram string to an image using `mermaid-cli`.

    Requires `mermaid-cli` to be installed manually. You can do that with:

        npm install -g @mermaid-js/mermaid-cli

    Args:
        diagram: String containing Mermaid diagram content.
        output_file: Path where output image should be saved. The extension dictates the
            file type.

    Raises:
        FileNotFoundError: If `mmdc` command is not found.
        subprocess.CalledProcessError: If conversion fails, or any other error from
            `mmdc`.
    """
    try:
        subprocess.run(
            ["mmdc", "-i", "-", "-o", str(output_file), "-w", "3200", "-H", "2400"],
            input=diagram.encode(),
            check=True,
        )
    except FileNotFoundError:
        print(
            "mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli"
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"mermaid-cli failed with error:\n{e.stderr.decode() if e.stderr else e}")
        raise


if __name__ == "__main__":
    app()
