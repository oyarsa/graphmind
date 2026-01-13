"""Summarise evaluation experiment results in a table.

This script scans experiment directories for metrics.json and params.json files,
then displays a formatted table with key metrics for comparison.

Usage:
    uv run python scripts/summarise_experiments.py output/eval_orc
    uv run python scripts/summarise_experiments.py output/eval_orc --filter ablation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table


def load_experiment(exp_dir: Path) -> dict[str, str | float] | None:
    """Load experiment metrics and params from a directory.

    Args:
        exp_dir: Path to experiment directory.

    Returns:
        Dictionary with experiment info, or None if files missing.
    """
    metrics_path = exp_dir / "metrics.json"
    params_path = exp_dir / "params.json"

    if not metrics_path.exists() or not params_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    with open(params_path) as f:
        params = json.load(f)

    return {
        "name": exp_dir.name,
        "prompt": params.get("eval_prompt_key", "N/A"),
        "model": params.get("model", "N/A"),
        "pearson": metrics.get("pearson", 0.0),
        "spearman": metrics.get("spearman", 0.0),
        "mae": metrics.get("mae", 0.0),
        "rmse": metrics.get("rmse", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "accuracy_within_1": metrics.get("accuracy_within_1", 0.0),
        "cost": metrics.get("cost", 0.0),
    }


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    experiments_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing experiment subdirectories."),
    ],
    filter_pattern: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-f",
            help="Only show experiments whose name contains this string.",
        ),
    ] = None,
    sort_by: Annotated[
        str,
        typer.Option(
            "--sort",
            "-s",
            help="Sort by this metric (name, pearson, spearman, mae, cost).",
        ),
    ] = "name",
    descending: Annotated[
        bool,
        typer.Option(
            "--desc",
            "-d",
            help="Sort in descending order.",
        ),
    ] = False,
) -> None:
    """Display a summary table of experiment results."""
    if not experiments_dir.is_dir():
        Console().print(f"[red]Error:[/red] {experiments_dir} is not a directory")
        raise typer.Exit(1)

    # Find all experiment directories
    experiments: list[dict[str, str | float]] = []
    for subdir in sorted(experiments_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if filter_pattern and filter_pattern not in subdir.name:
            continue

        exp_data = load_experiment(subdir)
        if exp_data:
            experiments.append(exp_data)

    if not experiments:
        Console().print("[yellow]No experiments found[/yellow]")
        raise typer.Exit(0)

    # Sort experiments
    if sort_by in (
        "pearson",
        "spearman",
        "mae",
        "cost",
        "accuracy",
        "accuracy_within_1",
    ):
        experiments.sort(key=lambda x: float(x[sort_by]), reverse=descending)
    else:
        experiments.sort(key=lambda x: str(x["name"]), reverse=descending)

    # Build table
    table = Table(title=f"Experiments in {experiments_dir}", expand=False)
    table.add_column("Experiment", style="cyan", no_wrap=True, min_width=20)
    table.add_column("Prompt", style="blue", no_wrap=True, min_width=22)
    table.add_column("Pears", justify="right")
    table.add_column("Spear", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("±1", justify="right")
    table.add_column("Exact", justify="right")
    table.add_column("Cost", justify="right", style="green")

    for exp in experiments:
        table.add_row(
            str(exp["name"]),
            str(exp["prompt"]),
            f"{float(exp['pearson']):.3f}",
            f"{float(exp['spearman']):.3f}",
            f"{float(exp['mae']):.2f}",
            f"{float(exp['accuracy_within_1']):.1%}",
            f"{float(exp['accuracy']):.1%}",
            f"${float(exp['cost']):.2f}",
        )

    console = Console(force_terminal=True, width=120)
    console.print(table)
    console.print(f"\n[dim]Found {len(experiments)} experiments[/dim]")


if __name__ == "__main__":
    app()
