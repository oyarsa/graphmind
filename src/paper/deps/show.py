"""Build dependency graph from dependency file using Mermaid."""

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
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


_DEPENDENCIES_DATA = read_resource("deps", "deps.yaml")


@app.command(help=__doc__)
def main(
    output_file: Annotated[Path, typer.Argument(help="File to save the graph.")],
    input_file: Annotated[
        Path | None,
        typer.Argument(help="Dependecy YAML file. Defaults to `paper.deps.deps.yaml`."),
    ] = None,
) -> None:
    data = yaml.safe_load(input_file.read_text() if input_file else _DEPENDENCIES_DATA)

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

    root_nodes, leaf_nodes = _find_roots_and_leaves(deps)
    print("\nRoot nodes (no parents):")
    for node in sorted(root_nodes):
        print(f"- {node}")

    print("\nLeaf nodes (no childer):")
    for node in sorted(leaf_nodes):
        print(f"- {node}")

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

        lines.append(
            f'    {src_id}["{dep.source}"]-->|"{dep.detail}"|{tgt_id}["{dep.target}"]'
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
