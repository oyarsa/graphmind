"""Create table with the results of the citation context classification experiments."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table


def parse_metrics(base_path: Path) -> list[dict[str, Any]]:
    metrics_data: list[dict[str, Any]] = []

    for file_path in base_path.rglob("metrics.json"):
        metrics: dict[str, Any] = json.loads(file_path.read_bytes())

        run_name = str(file_path.parent.relative_to(base_path))
        prompt, context_flag, model = run_name.split("_", maxsplit=2)
        run_info = {
            "prompt": prompt,
            "context": "extended" if context_flag.startswith("--use") else "original",
            "model": model,
        }

        metrics_data.append(run_info | metrics)

    return metrics_data


def create_table(metrics_data: Sequence[dict[str, Any]]) -> Table:
    table = Table(title="Metrics Comparison")

    table.add_column("Prompt", style="green")
    table.add_column("Context", style="yellow")
    table.add_column("Model", style="blue")

    for key in metrics_data[0]:
        if key not in ["prompt", "context", "model"]:
            table.add_column(key.capitalize(), style="magenta")

    for entry in metrics_data:
        row = [entry["prompt"], str(entry["context"]), entry["model"]]
        for key, value in entry.items():
            if key not in ["prompt", "context", "model"]:
                row.append(f"{value:.4f}")
        table.add_row(*row)

    return table


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    base_path: Annotated[Path, typer.Argument(help="Base path of the experiments")],
):
    metrics_data = parse_metrics(base_path)
    table = create_table(metrics_data)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    app()
