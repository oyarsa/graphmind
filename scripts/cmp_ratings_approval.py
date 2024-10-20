"""Compare classification metrics between ratings evaluation strategies and approval.

The approval is given in the ASAP dataset, but it's binary. We'll eventually want to
use the numerical ratings, so I want to know how well the approval matches the ratings.

For now, I'm not comparing the integer ratings with the approval, just the two strategies
I use to convert the ratings to a binary outcome.
"""

# pyright: basic
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, TypeAdapter
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from paper_hypergraph.gpt.model import RatingEvaluationStrategy


class InputEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    ratings: Sequence[int]
    approval: bool


class Approvals(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: bool
    majority: bool
    gold: bool


def calculate_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[float, float, float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return accuracy, precision, recall, f1


def print_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], strategy_name: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    table = Table("Gold\\Pred", "True", "False", title=f"\n{strategy_name} vs Gold")
    table.add_row("True", str(cm[1, 1]), str(cm[1, 0]))
    table.add_row("False", str(cm[0, 1]), str(cm[0, 0]))

    Console().print(table)


def print_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], strategy_name: str
) -> None:
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)

    table = Table("Metric", "Value", title=f"\n{strategy_name} vs Gold")
    table.add_row("Accuracy", f"{accuracy:.4f}")
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("F1", f"{f1:.4f}")

    Console().print(table)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[Path, typer.Argument(help="Path to input JSON file.")],
) -> None:
    data = TypeAdapter(list[InputEntry]).validate_json(input_file.read_bytes())

    approvals = [
        Approvals(
            mean=RatingEvaluationStrategy.MEAN.is_approved(d.ratings),
            majority=RatingEvaluationStrategy.MAJORITY.is_approved(d.ratings),
            gold=d.approval,
        )
        for d in data
    ]

    approvals_mean = [a.mean for a in approvals]
    approvals_majority = [a.majority for a in approvals]
    approvals_gold = [a.gold for a in approvals]

    print_metrics(approvals_gold, approvals_mean, "Mean")
    print_confusion_matrix(approvals_gold, approvals_mean, "Mean")

    print_metrics(approvals_gold, approvals_majority, "Majority")
    print_confusion_matrix(approvals_gold, approvals_majority, "Majority")


if __name__ == "__main__":
    app()
