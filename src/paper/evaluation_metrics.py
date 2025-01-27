"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from paper.util import metrics


class TargetMode(Enum):
    """Whether the target variable is an int (1-5) rating, or binary."""

    INT = "int"
    BIN = "bin"


class Metrics(BaseModel):
    """Classification and regression metrics."""

    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    f1: float
    accuracy: float
    mae: float
    mse: float
    correlation: float | None
    confusion: list[list[int]]
    mode: TargetMode

    def __str__(self) -> str:
        """Display metrics (P/R/F1/Acc), one per line, then the confusion matrix.

        If `mode` is `TargetMode.INT`, also shows MAE, MSE and Pearson correlation.
        """
        out = [
            f"Precision  : {self.precision:.4f}",
            f"Recall     : {self.recall:.4f}",
            f"F1         : {self.f1:.4f}",
            f"Accuracy   : {self.accuracy:.4f}",
        ]

        if self.mode is TargetMode.INT:
            corr = f"{self.correlation:.4f}" if self.correlation is not None else "N/A"
            out.extend(
                [
                    f"MAE        : {self.mae:.4f}",
                    f"MSE        : {self.mse:.4f}",
                    f"Correlation: {corr}",
                ]
            )

        out.extend(
            [
                "Confusion Matrix:",
                self._format_confusion(),
            ]
        )

        return "\n".join(out)

    def _format_confusion(self) -> str:
        """Format confusion matrix as a string with row and column labels."""
        n = len(self.confusion)

        if self.mode is TargetMode.BIN:
            labels = [0, 1]
        else:
            labels = range(1, 6)

        label_strs = [str(label) for label in labels]

        margin = 3
        col_label_padding = 8  # Space before column numbers
        cell_width = 4  # Width for numbers

        matrix_str = [
            " " * (col_label_padding + cell_width + 2) + "Predicted",
            " " * margin
            + " " * col_label_padding
            + "".join(f"{label:>{cell_width}}" for label in label_strs),
            "-" * (margin + col_label_padding + cell_width * n + 3),
        ]

        for i, row in enumerate(self.confusion):
            row_str = (
                " " * margin
                + f"True {label_strs[i]} |"
                + "".join(f"{cell:>{cell_width}}" for cell in row)
            )
            matrix_str.append(row_str)

        return "\n".join(matrix_str)


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    """Calculate classification metrics for multi-class classification (labels 1-5).

    Args:
        y_true: Ground truth labels (values 1-5)
        y_pred: Predicted labels (values 1-5)

    Returns:
        Metrics object containing macro-averaged precision, recall, F1 score, accuracy,
        Mean Absolute Error, Mean Squared Error, Pearson Correlation and the
        classification confusion matrix.

    Raises:
        ValueError: If input sequences are empty or of different lengths
    """
    if not y_true or not y_pred:
        raise ValueError("Input sequences cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have the same length")

    values = set(y_true) | set(y_pred)
    if values == {0, 1}:
        mode = TargetMode.BIN
        labels = [0, 1]
    else:
        mode = TargetMode.INT
        labels = range(1, 6)

    return Metrics(
        precision=metrics.precision(y_true, y_pred),
        recall=metrics.recall(y_true, y_pred),
        f1=metrics.f1_score(y_true, y_pred),
        accuracy=metrics.accuracy(y_true, y_pred),
        mae=metrics.mean_absolute_error(y_true, y_pred),
        mse=metrics.mean_squared_error(y_true, y_pred),
        correlation=metrics.pearson_correlation(y_true, y_pred),
        confusion=metrics.confusion_matrix(y_true, y_pred, labels=labels),
        mode=mode,
    )
