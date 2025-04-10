"""Run NovaScore experiment grid search."""
# ruff: noqa: RUF001 (allow 'ambiguous' characters, like alpha)

import itertools
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Parameter combinations to test
SIM = [
    0.6,
    0.8,
]
SCORE = [
    0.3,
    0.5,
    0.7,
    0.9,
]
ALPHA = [
    0,
    0.5,
]
BETA = [
    0.5,
    1,
]
GAMMA = [1.0]  # Default value, not varied in grid search


@dataclass
class RunResult:
    """Result of a single evaluation run."""

    sim_threshold: float
    score_threshold: float
    alpha: float
    beta: float
    gamma: float
    accuracy: float

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"sim={self.sim_threshold:.1f}, score={self.score_threshold:.1f}, "
            f"α={self.alpha:.1f}, β={self.beta:.1f}, γ={self.gamma:.1f} → "
            f"accuracy={self.accuracy:.4f}"
        )


def run_evaluation(
    results_path: Path,
    output_dir: Path,
    sim_threshold: float,
    score_threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
    limit: int = 0,
) -> float | None:
    """Run the evaluation command with the given parameters and return the accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # fmt: off
    cmd = [
        "uv", "run", "paper", "baselines", "nova", "evaluate",
        "--results", results_path,
        "--output", output_dir,
        "--limit", limit,
        "--sim-threshold", sim_threshold,
        "--score-threshold", score_threshold,
        "--alpha", alpha,
        "--beta", beta,
        "--gamma", gamma,
        "--no-save",
    ]
    # fmt: on

    try:
        subprocess.run(list(map(str, cmd)), check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Command output: {e.stdout.decode()}")
        print(f"Command error: {e.stderr.decode()}")
        return None

    try:
        metrics = json.loads((output_dir / "metrics.json").read_text())
        return metrics.get("accuracy")
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        return None


def print_results_table(results: list[RunResult]) -> None:
    """Print a formatted table of results using Rich."""
    console = Console()

    table = Table(title="Grid Search Results")

    # Add columns
    table.add_column("Sim Threshold", justify="center")
    table.add_column("Score Threshold", justify="center")
    table.add_column("Alpha (α)", justify="center")
    table.add_column("Beta (β)", justify="center")
    table.add_column("Gamma (γ)", justify="center")
    table.add_column("Accuracy", justify="right")

    # Sort by accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x.accuracy, reverse=True)
    for result in sorted_results:
        table.add_row(
            f"{result.sim_threshold:.1f}",
            f"{result.score_threshold:.1f}",
            f"{result.alpha:.1f}",
            f"{result.beta:.1f}",
            f"{result.gamma:.1f}",
            f"{result.accuracy:.4f}" if result.accuracy >= 0 else "FAILED",
        )

    console.print(table)

    # Print the best result
    if sorted_results and sorted_results[0].accuracy >= 0:
        best = sorted_results[0]
        console.print("\n[bold green]Best Parameters:[/bold green]")
        console.print(f"  Sim Threshold: {best.sim_threshold:.1f}")
        console.print(f"  Score Threshold: {best.score_threshold:.1f}")
        console.print(f"  Alpha (α): {best.alpha:.1f}")
        console.print(f"  Beta (β): {best.beta:.1f}")
        console.print(f"  Gamma (γ): {best.gamma:.1f}")
        console.print(f"  [bold]Accuracy: {best.accuracy:.4f}[/bold]")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command()
def main(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the results file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="Output file with results.")
    ],
    limit: Annotated[int, typer.Option("--limit", help="Limit parameter")] = 0,
) -> None:
    """Run a grid search over predefined parameter combinations."""
    console = Console()

    combinations = list(itertools.product(SIM, SCORE, ALPHA, BETA, GAMMA))
    total_runs = len(combinations)
    console.print(
        f"[bold]Running grid search with {total_runs} parameter combinations...[/bold]"
    )

    results: list[RunResult] = []

    # Run evaluation for each combination
    for i, (sim, score, alpha, beta, gamma) in enumerate(tqdm(combinations), 1):
        console.print(
            f"[bold]Run {i}/{total_runs}[/bold]: sim={sim:.1f}, score={score:.1f},"
            f" α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}",
            end=", ",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            accuracy = run_evaluation(
                input_file,
                Path(tmpdir),
                sim,
                score,
                alpha,
                beta,
                gamma,
                limit,
            )

        # Record result, use -1 for failed runs
        results.append(
            RunResult(
                sim_threshold=sim,
                score_threshold=score,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                accuracy=accuracy if accuracy is not None else -1.0,
            )
        )

        console.print(
            f"acc={accuracy:.4f}" if accuracy is not None else "[red]error[/red]"
        )

    # Print table of all results and best parameters
    print_results_table(results)
    output_file.write_text(json.dumps([asdict(r) for r in results], indent=2))


if __name__ == "__main__":
    app()
