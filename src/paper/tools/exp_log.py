"""Experiment log management tool.

Usage:
    uv run paper tools explog stats
    uv run paper tools explog stats --since 2025-01-14
    uv run paper tools explog stats --type ablation
    uv run paper tools explog summary
    uv run paper tools explog summary --since 2025-01-14
    uv run paper tools explog update
    uv run paper tools explog update --output output/eval_orc
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from paper.types import Immutable

app = typer.Typer(
    help="Experiment log management tool",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)
console = Console()

DEFAULT_LOG_PATH = Path("EXPERIMENT_LOG.yaml")
DEFAULT_OUTPUT_DIR = Path("output")


class MetricStats(Immutable):
    """Statistics for a single metric across runs."""

    mean: float
    stdev: float | None  # None for single-run experiments (e.g., baselines)
    min: float
    max: float


class ExperimentMetrics(Immutable):
    """Metrics recorded for an experiment."""

    pearson: MetricStats
    spearman: MetricStats
    mae: MetricStats
    accuracy: MetricStats
    f1: MetricStats
    cost_per_run: float


class ExperimentParameters(Immutable):
    """Parameters used for an experiment."""

    dataset: str
    model: str
    runs: int
    # GPT experiment fields
    eval_prompt: str | None = None
    sources: str | None = None
    demos: str | None = None
    # Baseline experiment fields
    method: str | None = None
    sim_threshold: float | None = None


class ExperimentRecord(Immutable):
    """A single experiment record from the log."""

    date: date
    name: str
    description: str
    reason: str
    command: str
    parameters: ExperimentParameters
    metrics: ExperimentMetrics
    total_cost: float
    conclusion: str
    type: str | None = None  # ablation, prompt_engineering, or baseline


class ExperimentLog(Immutable):
    """The full experiment log."""

    experiments: Sequence[ExperimentRecord]


def load_experiment_log(path: Path) -> ExperimentLog:
    """Load and parse the experiment log YAML file."""
    data = yaml.safe_load(path.read_bytes())
    return ExperimentLog(experiments=data["experiments"])


def find_experiment_dirs(output_dir: Path) -> set[Path]:
    """Find all experiment directories with metrics.json or run_*/metrics.json."""
    experiment_dirs: set[Path] = set()

    for metrics_file in output_dir.rglob("metrics.json"):
        parent = metrics_file.parent
        if parent.name.startswith("run_"):
            experiment_dirs.add(parent.parent)
        else:
            experiment_dirs.add(parent)

    return experiment_dirs


def get_experiment_name_from_path(path: Path) -> str:
    """Extract a canonical experiment name from its path."""
    parts = path.parts
    if "eval_orc" in parts:
        idx = parts.index("eval_orc")
        return "orc/" + "/".join(parts[idx + 1 :])
    if "eval_peerread" in parts:
        idx = parts.index("eval_peerread")
        return "peerread/" + "/".join(parts[idx + 1 :])
    return str(path)


def get_possible_log_names(exp_dir: Path) -> list[str]:
    """Generate possible log entry names for an experiment directory.

    The log uses various naming conventions:
    - Direct directory name: "ablation_sans"
    - Without ablation_ prefix: "sans"
    - With dataset suffix: "sans_orc", "sans_peerread"
    - Without _graph suffix: "norel" for "norel_graph"
    """
    dir_name = exp_dir.name
    base_name = dir_name.replace("ablation_", "")
    is_orc = "eval_orc" in exp_dir.parts
    suffix = "_orc" if is_orc else "_peerread"

    return [
        dir_name,
        base_name,
        f"{dir_name}{suffix}",
        f"{base_name}{suffix}",
        base_name.replace("_graph", ""),
        f"{base_name.replace('_graph', '')}{suffix}",
    ]


def is_experiment_logged(exp_dir: Path, logged_names: set[str]) -> bool:
    """Check if an experiment directory is already in the log."""
    return any(name in logged_names for name in get_possible_log_names(exp_dir))


def parse_date(value: str | None) -> date | None:
    """Parse a date string in YYYY-MM-DD format."""
    if value is None:
        return None
    return date.fromisoformat(value)


@app.command()
def stats(
    log_path: Annotated[
        Path, typer.Option("--log", help="Path to experiment log YAML")
    ] = DEFAULT_LOG_PATH,
    since: Annotated[
        str | None, typer.Option(help="Show experiments from this date (YYYY-MM-DD)")
    ] = None,
    until: Annotated[
        str | None, typer.Option(help="Show experiments up to this date (YYYY-MM-DD)")
    ] = None,
    exp_type: Annotated[
        str | None,
        typer.Option(
            "--type", help="Filter by type (ablation, prompt_engineering, baseline)"
        ),
    ] = None,
) -> None:
    """Show experiment statistics by day."""
    since_date = parse_date(since)
    until_date = parse_date(until)

    log = load_experiment_log(log_path)

    experiments = log.experiments
    if since_date:
        experiments = tuple(e for e in experiments if e.date >= since_date)
    if until_date:
        experiments = tuple(e for e in experiments if e.date <= until_date)
    if exp_type:
        experiments = tuple(e for e in experiments if e.type == exp_type)

    if not experiments:
        console.print("[yellow]No experiments found matching the filters.[/]")
        return

    by_date: dict[date, list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        by_date[exp.date].append(exp)

    title = "Experiment Statistics by Day"
    if exp_type:
        title += f" (type: {exp_type})"

    table = Table(title=title)
    table.add_column("Date", style="cyan")
    table.add_column("Experiments", justify="right", style="green")
    table.add_column("Cost", justify="right", style="yellow")

    total_count = 0
    total_cost = 0.0

    for exp_date in sorted(by_date.keys()):
        day_experiments = by_date[exp_date]
        day_count = len(day_experiments)
        day_cost = sum(e.total_cost for e in day_experiments)

        table.add_row(str(exp_date), str(day_count), f"${day_cost:.2f}")

        total_count += day_count
        total_cost += day_cost

    table.add_section()
    table.add_row(
        "[bold]Total[/]", f"[bold]{total_count}[/]", f"[bold]${total_cost:.2f}[/]"
    )

    console.print(table)


@app.command()
def summary(
    log_path: Annotated[
        Path, typer.Option("--log", help="Path to experiment log YAML")
    ] = DEFAULT_LOG_PATH,
    since: Annotated[
        str | None, typer.Option(help="Show experiments from this date (YYYY-MM-DD)")
    ] = None,
    until: Annotated[
        str | None, typer.Option(help="Show experiments up to this date (YYYY-MM-DD)")
    ] = None,
) -> None:
    """Show experiment summary by type."""
    since_date = parse_date(since)
    until_date = parse_date(until)

    log = load_experiment_log(log_path)

    experiments = log.experiments
    if since_date:
        experiments = tuple(e for e in experiments if e.date >= since_date)
    if until_date:
        experiments = tuple(e for e in experiments if e.date <= until_date)

    if not experiments:
        console.print("[yellow]No experiments found matching the filters.[/]")
        return

    by_type: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        exp_type = exp.type or "unknown"
        by_type[exp_type].append(exp)

    title = "Experiment Summary by Type"
    if since_date or until_date:
        date_range: list[str] = []
        if since_date:
            date_range.append(f"from {since_date}")
        if until_date:
            date_range.append(f"to {until_date}")
        title += f" ({', '.join(date_range)})"

    table = Table(title=title)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Cost", justify="right", style="yellow")

    total_count = 0
    total_cost = 0.0

    for exp_type in sorted(by_type.keys()):
        type_experiments = by_type[exp_type]
        type_count = len(type_experiments)
        type_cost = sum(e.total_cost for e in type_experiments)

        table.add_row(exp_type, str(type_count), f"${type_cost:.2f}")

        total_count += type_count
        total_cost += type_cost

    table.add_section()
    table.add_row(
        "[bold]Total[/]", f"[bold]{total_count}[/]", f"[bold]${total_cost:.2f}[/]"
    )

    console.print(table)


def get_experiment_mtime(exp_dir: Path) -> date | None:
    """Get the modification time of an experiment directory."""
    for candidate in [exp_dir / "run_0" / "params.json", exp_dir / "params.json"]:
        if candidate.exists():
            return datetime.fromtimestamp(candidate.stat().st_mtime, tz=UTC).date()
    return None


@app.command()
def update(
    output_dir: Annotated[
        Path, typer.Option("--output", help="Output directory to scan")
    ] = DEFAULT_OUTPUT_DIR,
    log_path: Annotated[
        Path, typer.Option("--log", help="Path to experiment log YAML")
    ] = DEFAULT_LOG_PATH,
    since: Annotated[
        str | None,
        typer.Option(help="Only check experiments from this date (YYYY-MM-DD)"),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(help="Only check experiments up to this date (YYYY-MM-DD)"),
    ] = None,
) -> None:
    """Check for experiments not in the log."""
    since_date = parse_date(since)
    until_date = parse_date(until)

    log = load_experiment_log(log_path)
    logged_names = {exp.name for exp in log.experiments}

    eval_orc = output_dir / "eval_orc"
    eval_peerread = output_dir / "eval_peerread"

    missing: list[tuple[Path, str, date | None]] = []

    for eval_dir in [eval_orc, eval_peerread]:
        if not eval_dir.exists():
            continue

        experiment_dirs = find_experiment_dirs(eval_dir)
        for exp_dir in sorted(experiment_dirs):
            exp_date = get_experiment_mtime(exp_dir)

            if since_date and exp_date and exp_date < since_date:
                continue
            if until_date and exp_date and exp_date > until_date:
                continue

            if not is_experiment_logged(exp_dir, logged_names):
                canonical_name = get_experiment_name_from_path(exp_dir)
                missing.append((exp_dir, canonical_name, exp_date))

    if not missing:
        console.print("[green]All experiments are logged![/]")
        return

    table = Table(title=f"Experiments Not in Log ({len(missing)} found)")
    table.add_column("Date", style="magenta")
    table.add_column("Path", style="cyan")
    table.add_column("Canonical Name", style="yellow")

    for path, name, exp_date in missing:
        date_str = str(exp_date) if exp_date else "?"
        table.add_row(date_str, str(path.relative_to(output_dir)), name)

    console.print(table)
    console.print(f"\n[yellow]Found {len(missing)} experiments not in the log.[/]")
