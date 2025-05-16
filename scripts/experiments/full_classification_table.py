"""Display full classification metrics from existing results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from paper import gpt
from paper.evaluation_metrics import Metrics, calculate_metrics, display_metrics_row
from paper.types import Immutable
from paper.util import safediv, seqcat
from paper.util.serde import load_data


class BinaryConfusionMatrix(Immutable):
    """Binary confusion matrix with TP, FP, TN, FN counts."""

    tp: int  # True Positives
    fp: int  # False Positives
    tn: int  # True Negatives
    fn: int  # False Negatives

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / (TP + TN + FP + FN)."""
        return safediv(self.tp + self.tn, self.total)

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        return safediv(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        return safediv(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        """F1 Score: 2 * (precision * recall) / (precision + recall)."""
        return safediv(2 * self.precision * self.recall, self.precision + self.recall)

    @property
    def negative_precision(self) -> float:
        """Negative Precision: TN / (TN + FN)."""
        return safediv(self.tn, self.tn + self.fn)

    @property
    def negative_recall(self) -> float:
        """Negative Recall: TN / (TN + FP)."""
        return safediv(self.tn, self.tn + self.fp)

    @property
    def negative_f1(self) -> float:
        """Negative F1 Score."""
        return safediv(
            2 * self.negative_precision * self.negative_recall,
            self.negative_precision + self.negative_recall,
        )

    @property
    def macro_precision(self) -> float:
        """Macro-averaged Precision."""
        return (self.precision + self.negative_precision) / 2

    @property
    def macro_recall(self) -> float:
        """Macro-averaged Recall."""
        return (self.recall + self.negative_recall) / 2

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 Score."""
        return (self.f1 + self.negative_f1) / 2

    @classmethod
    def from_predictions(
        cls, y_true: Sequence[int], y_pred: Sequence[int]
    ) -> BinaryConfusionMatrix:
        """Create confusion matrix from binary predictions.

        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)

        Returns:
            BinaryConfusionMatrix with counts of TP, FP, TN, FN

        Raises:
            ValueError: If inputs contain values other than 0 and 1
        """
        if not all(y in (0, 1) for y in seqcat(y_true, y_pred)):
            raise ValueError("Inputs must contain only binary values (0 or 1)")

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        return cls(tp=tp, fp=fp, tn=tn, fn=fn)

    def to_metrics(self) -> tuple[Metrics, Metrics, Metrics]:
        """Convert to regular, negative, and macro Metrics objects."""
        # Create y_true and y_pred sequences that would produce this confusion matrix
        y_true = [1] * self.tp + [0] * self.fp + [0] * self.tn + [1] * self.fn
        y_pred = [1] * self.tp + [1] * self.fp + [0] * self.tn + [0] * self.fn

        regular = calculate_metrics(y_true, y_pred, average="binary")

        # For negative metrics, flip the labels
        flipped_y_true = [1 - y for y in y_true]
        flipped_y_pred = [1 - y for y in y_pred]
        negative = calculate_metrics(flipped_y_true, flipped_y_pred, average="binary")

        # For macro metrics
        macro = calculate_metrics(y_true, y_pred, average="macro")

        return regular, negative, macro

    def __str__(self) -> str:
        """Format confusion matrix as a string."""
        return (
            f"Confusion Matrix (Total: {self.total}):\n"
            f"             Predicted\n"
            f"             Pos    Neg\n"
            f"True Pos  |  {self.tp:<4}  {self.fn:<4}\n"
            f"True Neg  |  {self.fp:<4}  {self.tn:<4}\n\n"
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1: {self.f1:.4f}\n"
        )


def display_binary_confusion_matrix(matrix: BinaryConfusionMatrix) -> str:
    """Display binary confusion matrix with metrics.

    Args:
        matrix: Binary confusion matrix
        format: Display format - 'full' for detailed output or 'compact' for just the table

    Returns:
        Formatted string representation
    """

    regular, negative, macro = matrix.to_metrics()

    output = [
        f"Binary Confusion Matrix (Total: {matrix.total}):",
        "",
        "             Predicted",
        "             Pos    Neg",
        f"True Pos  |  {matrix.tp:<4}  {matrix.fn:<4}",
        f"True Neg  |  {matrix.fp:<4}  {matrix.tn:<4}",
        "",
        "> Positive:",
        f"Precision  : {matrix.precision:.4f}",
        f"Recall     : {matrix.recall:.4f}",
        f"F1         : {matrix.f1:.4f}",
        "",
        "> Negative:",
        f"Precision  : {matrix.negative_precision:.4f}",
        f"Recall     : {matrix.negative_recall:.4f}",
        f"F1         : {matrix.negative_f1:.4f}",
        "",
        "> Macro-averaged:",
        f"Precision  : {matrix.macro_precision:.4f}",
        f"Recall     : {matrix.macro_recall:.4f}",
        f"F1         : {matrix.macro_f1:.4f}",
        f"Accuracy   : {matrix.accuracy:.4f}",
        "",
        "> Table:",
        display_metrics_row(regular, negative, macro),
    ]

    return "\n".join(output)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(no_args_is_help=True)
def confusion(
    tn: Annotated[int, typer.Argument(help="True negatives")],
    fn: Annotated[int, typer.Argument(help="False negatives")],
    fp: Annotated[int, typer.Argument(help="False positives")],
    tp: Annotated[int, typer.Argument(help="True positives")],
) -> None:
    """From a confusion matrix: TN/FN/FP/TP counts."""
    matrix = BinaryConfusionMatrix(tn=tn, fn=fn, fp=fp, tp=tp)
    print(display_binary_confusion_matrix(matrix))


@app.command(no_args_is_help=True)
def metrics(
    metrics_file: Annotated[Path, typer.Argument(help="Path to metrics file")],
) -> None:
    """From a metrics.json file.

    The file must be a JSON object with a `confusion` field, which must be a 2D array,
    the confusion matrix.
    """
    data: dict[str, Any] = json.loads(metrics_file.read_bytes())
    confusion = data["confusion"]

    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tp = confusion[1][1]

    matrix = BinaryConfusionMatrix(tn=tn, fn=fn, fp=fp, tp=tp)
    print(display_binary_confusion_matrix(matrix))


@app.command(no_args_is_help=True)
def results(
    results_file: Annotated[Path, typer.Argument(help="Path to results file")],
) -> None:
    """From a result.json file with format `gpt.PromptResult[gpt.GraphResult]`.

    I.e. the output of `gpt.evaluate_paper_graph`.
    """
    data = load_data(results_file, gpt.PromptResult[gpt.GraphResult])

    tn, fp, fn, tp = 0, 0, 0, 0

    for p in data:
        true_val = p.item.paper.y_true
        pred_val = p.item.paper.y_pred
        if true_val == 0 and pred_val == 0:
            tn += 1  # True Negative
        elif true_val == 0 and pred_val == 1:
            fp += 1  # False Positive
        elif true_val == 1 and pred_val == 0:
            fn += 1  # False Negative
        elif true_val == 1 and pred_val == 1:
            tp += 1  # True Positive

    matrix = BinaryConfusionMatrix(tn=tn, fn=fn, fp=fp, tp=tp)
    print(display_binary_confusion_matrix(matrix))


if __name__ == "__main__":
    app()
