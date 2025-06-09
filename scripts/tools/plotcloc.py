"""Show statistics about lines of code in Python files.

Takes a directory path as input and shows distribution of lines of code across Python
files.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backend_bases import Event, MouseEvent

if TYPE_CHECKING:
    from matplotlib.patches import Wedge

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    dir: Annotated[
        Path,
        typer.Argument(
            help="Directory with Python files to analyse.", exists=True, file_okay=False
        ),
    ] = Path(),
) -> None:
    """Analyse Python files and plot CLOC stats."""
    files = [
        FileStats(path=path, loc=_count_lines(path))
        for path in dir.rglob("*.py")
        if "env" not in str(path)
    ]
    if not files:
        sys.exit(f"No Python files found in '{dir}'")

    total_lines = sum(f.loc for f in files)
    shares = [
        Share(file=str(f.path), code=f.loc, pct=f.loc / total_lines) for f in files
    ]

    show_pie_chart(shares, str(dir))


@dataclass(frozen=True, kw_only=True)
class FileStats:
    """Python file path and its number of lines."""

    path: Path
    loc: int


def _count_lines(file_path: Path) -> int:
    """Count non-empty lines in a file."""
    try:
        return sum(bool(line.strip()) for line in file_path.read_text().splitlines())
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0


@dataclass(frozen=True, kw_only=True)
class Share:
    """How much a given file's number of lines contributes to the total."""

    file: str
    code: int
    pct: float


def show_pie_chart(shares: Sequence[Share], name: str) -> None:
    """Show an interactive pie chart of code distribution."""
    shares = sorted(shares, key=lambda x: x.pct, reverse=True)
    files = [share.file for share in shares]
    lines = [share.code for share in shares]
    percentages = [share.pct * 100 for share in shares]

    fig, ax = plt.subplots(figsize=(10, 8))  # type: ignore

    def autopct(pct: float) -> str:
        idx = next(i for i, p in enumerate(percentages) if abs(p - pct) < 0.1)
        return f"{lines[idx]:,d} ({pct:.1f}%)"

    patches = ax.pie(
        percentages,
        autopct=autopct,
        startangle=90,
        pctdistance=0.75,
    )[0]

    ax.axis("equal")

    annotation = ax.annotate(  # type: ignore
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "w", "ec": "0.5", "alpha": 0.9},
        ha="center",
    )
    annotation.set_visible(False)

    def hover(event: Event) -> None:
        assert isinstance(event, MouseEvent)

        if event.inaxes != ax:
            annotation.set_visible(False)
            plt.draw()
            return

        found = False
        for i, patch in enumerate(patches):
            contains, _ = patch.contains(event)
            if contains:
                show_annotation(patch, i)
                found = True
                break

        if not found:
            annotation.set_visible(False)
            plt.draw()

    def show_annotation(patch: Wedge, index: int) -> None:
        annotation.set_text(
            f"{files[index]} {lines[index]} ({percentages[index]:.1f}%)"
        )

        theta = np.pi / 2 - (patch.theta1 + patch.theta2) / 2
        r = patch.r
        center = (r / 2 * np.cos(theta), r / 2 * np.sin(theta))
        annotation.xy = center

        annotation.set_visible(True)
        plt.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title(f"Python Code Distribution - {name}")  # type: ignore
    plt.tight_layout()
    plt.show()  # type: ignore


if __name__ == "__main__":
    app()
