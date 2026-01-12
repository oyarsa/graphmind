"""Run evaluation experiments multiple times and aggregate metrics."""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single evaluation run."""

    run_idx: int
    output_dir: Path
    metrics: dict[str, Any]
    success: bool

    @property
    def pearson(self) -> float | None:
        """Pearson correlation coefficient."""
        return self.metrics.get("pearson")

    @property
    def spearman(self) -> float | None:
        """Spearman correlation coefficient."""
        return self.metrics.get("spearman")

    @property
    def cost(self) -> float | None:
        """Total cost of the run."""
        return self.metrics.get("cost")

    @property
    def mae(self) -> float | None:
        """Mean absolute error."""
        return self.metrics.get("mae")

    @property
    def accuracy(self) -> float | None:
        """Exact accuracy."""
        return self.metrics.get("accuracy")

    @property
    def f1(self) -> float | None:
        """Macro F1 score."""
        return self.metrics.get("f1")


@dataclass
class AggregateStats:
    """Aggregated statistics across multiple runs."""

    name: str
    values: list[float | None]

    @property
    def valid_values(self) -> list[float]:
        """Filter out None values."""
        return [v for v in self.values if v is not None]

    @property
    def n_valid(self) -> int:
        """Number of valid values."""
        return len(self.valid_values)

    @property
    def mean(self) -> float | None:
        """Mean of valid values."""
        return statistics.mean(self.valid_values) if self.valid_values else None

    @property
    def median(self) -> float | None:
        """Median of valid values."""
        return statistics.median(self.valid_values) if self.valid_values else None

    @property
    def stdev(self) -> float | None:
        """Standard deviation of valid values."""
        if len(self.valid_values) < 2:
            return None
        return statistics.stdev(self.valid_values)

    @property
    def min(self) -> float | None:
        """Minimum of valid values."""
        return min(self.valid_values) if self.valid_values else None

    @property
    def max(self) -> float | None:
        """Maximum of valid values."""
        return max(self.valid_values) if self.valid_values else None


@dataclass
class ExperimentResults:
    """Results from running multiple evaluation experiments."""

    runs: list[RunResult]
    output_dir: Path

    @property
    def successful_runs(self) -> list[RunResult]:
        """Return only successful runs."""
        return [r for r in self.runs if r.success]

    @property
    def n_successful(self) -> int:
        """Number of successful runs."""
        return len(self.successful_runs)

    def get_stats(self, metric: str) -> AggregateStats:
        """Get aggregate statistics for a metric."""
        values = [r.metrics.get(metric) for r in self.successful_runs]
        return AggregateStats(name=metric, values=values)

    @property
    def pearson_stats(self) -> AggregateStats:
        """Pearson correlation stats."""
        return self.get_stats("pearson")

    @property
    def spearman_stats(self) -> AggregateStats:
        """Spearman correlation stats."""
        return self.get_stats("spearman")

    @property
    def cost_stats(self) -> AggregateStats:
        """Cost stats."""
        return self.get_stats("cost")

    @property
    def mae_stats(self) -> AggregateStats:
        """MAE stats."""
        return self.get_stats("mae")

    @property
    def total_cost(self) -> float:
        """Total cost across all runs."""
        return sum(self.cost_stats.valid_values)

    def save_summary(self) -> Path:
        """Save summary to JSON file."""
        summary_path = self.output_dir / "experiment_summary.json"

        metrics_summary: dict[str, dict[str, Any]] = {}
        for metric in ["pearson", "spearman", "mae", "accuracy", "f1", "cost"]:
            stats = self.get_stats(metric)
            metrics_summary[metric] = {
                "values": stats.values,
                "mean": stats.mean,
                "median": stats.median,
                "stdev": stats.stdev,
                "min": stats.min,
                "max": stats.max,
            }

        summary: dict[str, Any] = {
            "n_runs": len(self.runs),
            "n_successful": self.n_successful,
            "total_cost": self.total_cost,
            "metrics": metrics_summary,
        }

        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path


def format_stat(value: float | None, fmt: str = ".4f") -> str:
    """Format a statistic value for display."""
    return f"{value:{fmt}}" if value is not None else "N/A"


def print_experiment_summary(results: ExperimentResults) -> None:
    """Print formatted summary of experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Successful runs: {results.n_successful}/{len(results.runs)}")
    print(f"Total cost: ${results.total_cost:.4f}")

    # Main correlation metrics
    print("\n--- Correlation Metrics ---")
    for name, stats in [
        ("Pearson", results.pearson_stats),
        ("Spearman", results.spearman_stats),
    ]:
        print(f"\n{name}:")
        print(f"  Mean:   {format_stat(stats.mean)}")
        print(f"  Median: {format_stat(stats.median)}")
        print(f"  Stdev:  {format_stat(stats.stdev)}")
        print(f"  Range:  [{format_stat(stats.min)}, {format_stat(stats.max)}]")

    # Other metrics
    print("\n--- Other Metrics ---")
    for name, stats in [
        ("MAE", results.mae_stats),
        ("Accuracy", results.get_stats("accuracy")),
        ("Macro F1", results.get_stats("f1")),
    ]:
        print(
            f"{name}: mean={format_stat(stats.mean)}, median={format_stat(stats.median)}"
        )

    # Per-run cost
    print("\n--- Cost per Run ---")
    print(f"Mean: ${format_stat(results.cost_stats.mean, '.6f')}")

    # Raw values for reference
    print("\n--- Raw Values ---")
    print(f"Pearson:  {results.pearson_stats.values}")
    print(f"Spearman: {results.spearman_stats.values}")


def run_single_experiment(
    cmd: Sequence[str],
    run_idx: int,
    output_dir: Path,
    n_runs: int,
) -> RunResult:
    """Run a single evaluation experiment.

    Args:
        cmd: Command to run (without --output flag).
        run_idx: Index of this run (0-based).
        output_dir: Directory for this run's output.
        n_runs: Total number of runs (for display).

    Returns:
        RunResult with metrics and success status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    full_cmd = [*cmd, "--output", str(output_dir)]

    print(f"\n{'=' * 60}")
    print(f"Run {run_idx + 1}/{n_runs}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    result = subprocess.run(full_cmd, check=False)

    metrics_file = output_dir / "metrics.json"
    if result.returncode != 0 or not metrics_file.exists():
        logger.warning(f"Run {run_idx + 1} failed (exit code {result.returncode})")
        return RunResult(
            run_idx=run_idx,
            output_dir=output_dir,
            metrics={},
            success=False,
        )

    metrics = json.loads(metrics_file.read_bytes())
    return RunResult(
        run_idx=run_idx,
        output_dir=output_dir,
        metrics=metrics,
        success=True,
    )


def run_experiment(
    base_cmd: Sequence[str],
    output_dir: Path,
    n_runs: int,
) -> ExperimentResults:
    """Run evaluation experiment multiple times.

    Args:
        base_cmd: Base command to run (without --output flag).
        output_dir: Base output directory. Each run saves to output_dir/run_N.
        n_runs: Number of runs.

    Returns:
        ExperimentResults with all run results and aggregate stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunResult] = []
    for i in range(n_runs):
        run_output = output_dir / f"run_{i}"
        result = run_single_experiment(base_cmd, i, run_output, n_runs)
        runs.append(result)

    results = ExperimentResults(runs=runs, output_dir=output_dir)

    # Save summary
    summary_path = results.save_summary()
    logger.info(f"Experiment summary saved to {summary_path}")

    return results


def build_eval_command(
    eval_type: str,
    *,
    papers: Path,
    model: str,
    limit: int,
    seed: int,
    n_evaluations: int,
    eval_temperature: float,
    batch_size: int,
    demos: str | None,
    demo_prompt: str,
    # Graph-specific
    eval_prompt: str | None = None,
    graph_prompt: str | None = None,
    linearisation: str | None = None,
    sources: list[str] | None = None,
    temperature: float | None = None,
    # Scimon-specific
    user_prompt: str | None = None,
) -> list[str]:
    """Build the evaluation command.

    Args:
        eval_type: Either 'graph' or 'scimon'.
        papers: Path to input papers file.
        model: Model to use.
        limit: Number of papers to process.
        seed: Random seed.
        n_evaluations: Number of evaluation rounds per paper.
        eval_temperature: Temperature for evaluation rounds.
        batch_size: Batch size for requests.
        demos: Demonstrations file name.
        demo_prompt: Demo prompt type.
        eval_prompt: Evaluation prompt (graph only).
        graph_prompt: Graph extraction prompt (graph only).
        linearisation: Linearisation method (graph only).
        sources: Related paper sources (graph only).
        temperature: Model temperature (graph only).
        user_prompt: User prompt (scimon only).

    Returns:
        Command as list of strings.
    """
    cmd = [
        "uv",
        "run",
        "paper",
        "gpt",
        "eval",
        eval_type,
        "run",
        "--papers",
        str(papers),
        "--model",
        model,
        "-n",
        str(limit),
        "--seed",
        str(seed),
        "--n-evaluations",
        str(n_evaluations),
        "--eval-temperature",
        str(eval_temperature),
        "--batch-size",
        str(batch_size),
        "--demo-prompt",
        demo_prompt,
    ]

    if demos:
        cmd.extend(["--demos", demos])

    if eval_type == "graph":
        if eval_prompt:
            cmd.extend(["--eval-prompt", eval_prompt])
        if graph_prompt:
            cmd.extend(["--graph-prompt", graph_prompt])
        if linearisation:
            cmd.extend(["--linearisation", linearisation])
        if sources:
            for source in sources:
                cmd.extend(["--sources", source])
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])

    elif eval_type == "scimon":
        if user_prompt:
            cmd.extend(["--user-prompt", user_prompt])

    return cmd
