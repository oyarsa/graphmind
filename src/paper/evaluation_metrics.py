"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from paper.util import metrics, safediv


class TargetMode(Enum):
    """Whether the target variable is an int (1-5) rating, or binary."""

    INT = "int"
    BIN = "bin"
    UNCERTAIN = "uncertain"

    def labels(self) -> list[int]:
        """Labels represented by this mode."""
        match self:
            case TargetMode.INT:
                return list(range(1, 6))
            case TargetMode.BIN:
                return [0, 1]
            case TargetMode.UNCERTAIN:
                return [0, 1, 2]


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
            out.extend([
                f"MAE        : {self.mae:.4f}",
                f"MSE        : {self.mse:.4f}",
                f"Correlation: {corr}",
            ])

        return "\n".join(out)

    def display_confusion(self) -> str:
        """Display confusion matrix between classes."""
        return (
            f"Confusion Matrix:\n{format_confusion(self.confusion, self.mode.labels())}"
        )


def format_confusion(
    confusion: list[list[int]],
    labels: Iterable[Any],
    margin: int = 3,
    col_padding: int = 8,
    cell_width: int = 4,
) -> str:
    """Format confusion matrix as a string with row and column labels."""
    n = len(confusion)

    label_strs = [str(label) for label in labels]

    matrix_str = [
        " " * (col_padding + cell_width + 2) + "Predicted",
        " " * margin
        + " " * col_padding
        + "".join(f"{label:>{cell_width}}" for label in label_strs),
        "-" * (margin + col_padding + cell_width * n + 3),
    ]

    for i, row in enumerate(confusion):
        row_str = (
            " " * margin
            + f"True {label_strs[i]} |"
            + "".join(f"{cell:>{cell_width}}" for cell in row)
        )
        matrix_str.append(row_str)

    return "\n".join(matrix_str)


class Evaluated(Protocol):
    """Object with `y_true` and `y_pred` fields."""

    @property
    def y_true(self) -> int:
        """Gold label."""
        ...

    @property
    def y_pred(self) -> int:
        """Predicted label."""
        ...


def calculate_paper_metrics(papers: Sequence[Evaluated], cost: float) -> PaperMetrics:
    """Calculate evaluation metrics, including how much it cost.

    See also `paper.evaluation_metrics.calculate_metrics`.
    """
    y_pred = [p.y_pred for p in papers]
    y_true = [p.y_true for p in papers]

    return PaperMetrics.from_eval(
        calculate_metrics(y_true, y_pred), cost, y_true, y_pred
    )


def display_metrics(metrics: Metrics, results: Sequence[Evaluated]) -> str:
    """Display metrics and distribution statistics from the results.

    `evaluation_metrics.Metrics` are displayed directly. The distribution statistics are
    shown as the count and percentage of true/false for both gold and prediction.

    Args:
        metrics: Metrics calculated using `evaluation_metrics.calculate_metrics`.
        results: Paper evaluation results.

    Returns:
        Formatted string showing both metrics and distribution statistics.
    """
    y_true = [r.y_true for r in results]
    y_pred = [r.y_pred for r in results]

    output = [
        "Metrics:",
        str(metrics),
        metrics.display_confusion(),
    ]
    for values, section in [(y_true, "Gold"), (y_pred, "Predicted")]:
        output.append(f"\n{section} distribution:")
        for label in metrics.mode.labels():
            count = sum(y == label for y in values)
            output.append(
                f"  {label}: {count}/{len(values)} ({safediv(count, len(values)):.2%})"
            )
    return "\n".join(output)


class RatingStats(BaseModel):
    """Mean/stdev/median stats on novelty ratings."""

    model_config = ConfigDict(frozen=True)

    mean: float
    stdev: float
    median: float

    @classmethod
    def calc(cls, values: Sequence[int]) -> RatingStats:
        """Calculate stats from sequence of values."""
        import statistics

        return cls(
            mean=statistics.mean(values),
            stdev=statistics.stdev(values) if len(values) > 2 else 0,
            median=statistics.median(values),
        )

    def __str__(self) -> str:
        """Format stats one per line."""
        return "\n".join([
            f"mean   : {self.mean:.4f}",
            f"stdev  : {self.stdev:.4f}",
            f"median : {self.median:.4f}",
        ])


class PaperMetrics(Metrics):
    """Evaluation metrics with total API cost."""

    cost: float
    stats_pred: RatingStats
    stats_true: RatingStats

    @classmethod
    def from_eval(
        cls,
        eval: Metrics,
        cost: float,
        y_true: Sequence[int],
        y_pred: Sequence[int],
    ) -> PaperMetrics:
        """Build metrics with cost from standard evaluation metrics."""
        return cls.model_validate(
            eval.model_dump()
            | {
                "cost": cost,
                "stats_pred": RatingStats.calc(y_pred),
                "stats_true": RatingStats.calc(y_true),
            }
        )


def _guess_target_mode(y_pred: Sequence[int], y_true: Sequence[int]) -> TargetMode:
    """Guess target mode from the possible values.

    If only 0 and 1 are possible in `y_pred` and `y_true`, the mode is BIN. If 0, 1 and
    2 are possible, it's UNCERTAIN (2 is uncertain). Otherwise, it's INT.

    This isn't always accurate. For example, if the data is binary but it's always 0 or
    1, this will incorrectly assume it's INT and not BIN since there's no way to tell.
    """
    values = set(y_true) | set(y_pred)
    if values == {0, 1}:
        return TargetMode.BIN
    if values == {0, 1, 2}:
        return TargetMode.UNCERTAIN
    else:
        return TargetMode.INT


def calculate_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], mode: TargetMode | None = None
) -> Metrics:
    """Calculate classification metrics for multi-class classification.

    The labels can be either in 1-5 or binary (0/1). This is determined by the range of
    values in the inputs.

    Args:
        y_true: Ground truth labels (values 1-5 or 0/1)
        y_pred: Predicted labels (values 1-5 or 0/1)
        mode: What mode are the labels, either BIN (0/1) or INT (1-5). If absent, we will
            attempt to find the mode by checking the possible values.

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

    if mode is None:
        mode = _guess_target_mode(y_pred, y_true)

    return Metrics(
        precision=metrics.precision(y_true, y_pred),
        recall=metrics.recall(y_true, y_pred),
        f1=metrics.f1_score(y_true, y_pred),
        accuracy=metrics.accuracy(y_true, y_pred),
        mae=metrics.mean_absolute_error(y_true, y_pred),
        mse=metrics.mean_squared_error(y_true, y_pred),
        correlation=metrics.pearson_correlation(y_true, y_pred),
        confusion=metrics.confusion_matrix(y_true, y_pred, labels=mode.labels()),
        mode=mode,
    )
