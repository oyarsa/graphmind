"""Create table with the results of the citation context classification experiments."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table


def parse_metrics(base_path: Path) -> list[dict[str, Any]]:
    """Parse metrics and results files from an experiment run."""
    metrics_data: list[dict[str, Any]] = []

    for dir in base_path.iterdir():
        if not dir.is_dir():
            continue

        metrics: dict[str, Any] = json.loads((dir / "metrics.json").read_bytes())
        results: list[dict[str, Any]] = json.loads((dir / "result.json").read_bytes())

        run_name = str(dir.relative_to(base_path))
        prompt, context_flag, model = run_name.split("_", maxsplit=2)
        run_info = {
            "prompt": prompt,
            "context": "extended" if context_flag.startswith("--use") else "original",
            "model": model,
            "n": str(count_contexts(results)),
        }

        metrics_data.append(run_info | metrics)

    return metrics_data


def count_contexts(data: Iterable[dict[str, Any]]) -> int:
    """Count total number of contexts across all papers and references."""
    return sum(
        len(reference["contexts"])
        for paper in data
        for reference in paper["item"]["references"]
    )


def create_table(metrics_data: Sequence[dict[str, Any]]) -> Table:
    """Create pretty table with metrics from an experiment run."""
    table = Table(title="Context classification results")

    info_columns = ["prompt", "context", "model", "n"]
    info_colours = ["green", "yellow", "blue", "red"]

    for col, colour in zip(info_columns, info_colours):
        table.add_column(col.capitalize(), style=colour)

    for key in metrics_data[0]:
        if key not in info_columns:
            table.add_column(key.capitalize(), style="magenta")

    for entry in sorted(metrics_data, key=lambda x: x["f1"], reverse=True):
        row = [entry[col] for col in info_columns]
        for key, value in entry.items():
            if key not in info_columns:
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
) -> None:
    """Show table with metrics from an experiment run."""
    metrics_data = parse_metrics(base_path)
    table = create_table(metrics_data)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    app()
