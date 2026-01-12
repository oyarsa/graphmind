"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from paper.types import Immutable
from paper.util import metrics, safediv

# Valid rating labels for novelty evaluation (1-5 scale)
RATING_LABELS = list(range(1, 6))


class Metrics(Immutable):
    """Classification and regression metrics for 1-5 ratings."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    mae: float
    mse: float
    rmse: float
    pearson: float | None
    spearman: float | None
    accuracy_within_1: float
    confusion: list[list[int]]
    confidence: float | None
    cost: float | None

    def __str__(self) -> str:
        """Display metrics for 1-5 ratings.

        Emphasises regression metrics (MAE, RMSE, correlations, accuracy within ±1).
        """
        pearson = f"{self.pearson:.4f}" if self.pearson is not None else "N/A"
        spearman = f"{self.spearman:.4f}" if self.spearman is not None else "N/A"
        out = [
            f"MAE            : {self.mae:.4f}",
            f"RMSE           : {self.rmse:.4f}",
            f"Accuracy (±1)  : {self.accuracy_within_1:.4f}",
            f"Pearson        : {pearson}",
            f"Spearman       : {spearman}",
            f"Exact Accuracy : {self.accuracy:.4f}",
            f"Macro F1       : {self.f1:.4f}",
        ]

        if self.confidence is not None:
            out.append(f"Confidence : {self.confidence:.4f}")

        return "\n".join(out)

    def display_confusion(self) -> str:
        """Display confusion matrix between classes."""
        return f"Confusion Matrix:\n{format_confusion(self.confusion, RATING_LABELS)}"


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


@runtime_checkable
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


@runtime_checkable
class EvaluatedWithConfidence(Protocol):
    """Object with `y_true`, `y_pred` and `confidence` fields."""

    @property
    def y_true(self) -> int:
        """Gold label."""
        ...

    @property
    def y_pred(self) -> int:
        """Predicted label."""
        ...

    @property
    def confidence(self) -> float | None:
        """Confidence of the prediction, between 0 and 1, or None if not available."""


def calculate_paper_metrics(
    papers: Sequence[Evaluated | EvaluatedWithConfidence],
    average: Literal["binary", "macro", "micro"] | None = None,
    cost: float | None = None,
) -> Metrics:
    """Calculate evaluation metrics.

    Args:
        papers: Papers to evaluate. The only requirement is having `y_pred` and `y_true`
            integer properties.
        average: What average mode to use for precision/recall/F1. If None, will use
            'binary' when mode is binary and 'macro' for everything else.
        cost: Cost associated with the predictions, if any.

    See also `paper.evaluation_metrics.calculate_metrics`.
    """
    y_pred = [p.y_pred for p in papers]
    y_true = [p.y_true for p in papers]
    confidences = [
        p.confidence
        for p in papers
        if isinstance(p, EvaluatedWithConfidence) and p.confidence is not None
    ]

    return calculate_metrics(
        y_true, y_pred, average=average, confidences=confidences, cost=cost
    )


def calculate_negative_paper_metrics(papers: Sequence[Evaluated]) -> Metrics:
    """Calculate evaluation metrics for the negative label in binary classification.

    Deprecated: This function was for binary classification and is not applicable
    for 1-5 integer ratings. Returns the same as regular metrics.
    """
    return calculate_paper_metrics(papers)


def display_metrics(metrics_obj: Metrics, results: Sequence[Evaluated]) -> str:
    """Display metrics and distribution statistics from the results.

    `evaluation_metrics.Metrics` are displayed directly. The distribution statistics are
    shown as the count and percentage of true/false for both gold and prediction.

    Args:
        metrics_obj: Metrics calculated using `evaluation_metrics.calculate_metrics`.
        results: Paper evaluation results.

    Returns:
        Formatted string showing both metrics and distribution statistics.
    """
    output = [
        "Metrics:",
        str(metrics_obj),
        metrics_obj.display_confusion(),
        "",
        display_metrics_distribution(results),
    ]
    return "\n".join(output)


def display_metrics_distribution(results: Sequence[Evaluated]) -> str:
    """Show distribution of true and predicted labels."""
    y_true = [r.y_true for r in results]
    y_pred = [r.y_pred for r in results]

    output: list[str] = []

    for values, section in [(y_true, "Gold"), (y_pred, "Predicted")]:
        output.append(f"\n{section} distribution:")
        for label in RATING_LABELS:
            count = sum(y == label for y in values)
            output.append(
                f"  {label}: {count}/{len(values)} ({safediv(count, len(values)):.2%})"
            )
    return "\n".join(output)


def display_regular_negative_macro_metrics(items: Sequence[Evaluated]) -> str:
    """Display evaluation metrics for 1-5 integer ratings.

    Shows key regression metrics (MAE, RMSE, correlations, accuracy within ±1)
    along with distribution statistics.
    """
    regular = calculate_paper_metrics(items)

    out = [
        str(regular),
        "",
        regular.display_confusion(),
        "",
    ]

    return "\n".join(out)


def display_metrics_row(
    regular: Metrics, negative: Metrics | None = None, macro: Metrics | None = None
) -> str:
    """Build a formatted markdown table displaying classification metrics.

    Parameters:
        regular: Regular (positive) metrics (always displayed).
        negative: Negative class metrics (optional).
        macro: Macro-averaged metrics (optional).

    Returns:
        A markdown-formatted table string with aligned columns.
    """
    header = ["Acc"]
    values = [regular.accuracy]

    for prefix, entry in [("P", regular), ("N", negative), ("M", macro)]:
        if entry:
            header += [f"{prefix}-P", f"{prefix}-R", f"{prefix}-F1", f"{prefix}-CO"]
            values += [
                entry.precision,
                entry.recall,
                entry.f1,
                entry.confidence or 1,
            ]

    values_txt = [f"{x:.4f}" for x in values]

    padding = [max(len(h), len(v)) + 1 for h, v in zip(header, values_txt)]

    header_padded = [f"{h:{pad}}" for h, pad in zip(header, padding)]
    sep_row = ["-" * pad for pad in padding]
    values_padded = [f"{v:{pad}}" for v, pad in zip(values_txt, padding)]

    return "\n".join(
        f"| {' | '.join(row)}|" for row in (header_padded, sep_row, values_padded)
    )


class RatingStats(Immutable):
    """Mean/stdev/median stats on novelty ratings."""

    mean: float
    stdev: float
    median: float

    @classmethod
    def calc(cls, values: Sequence[int]) -> RatingStats:
        """Calculate stats from sequence of values."""

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


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: Literal["binary", "macro", "micro"] | None = None,
    confidences: Sequence[float] | None = None,
    cost: float | None = None,
) -> Metrics:
    """Calculate classification metrics for 1-5 integer ratings.

    Args:
        y_true: Ground truth labels (values 1-5)
        y_pred: Predicted labels (values 1-5)
        average: What average mode to use for precision/recall/F1. Defaults to 'macro'.
        confidences: Novelty label confidence for each item, if present. If absent, the
            'confidence' output will be null.
        cost: Cost associated with the predictions, if any.

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

    if average is None:
        average = "macro"

    if confidences:
        confidence = sum(confidences) / len(confidences)
    else:
        confidence = 1

    return Metrics(
        precision=metrics.precision(y_true, y_pred, average=average),
        recall=metrics.recall(y_true, y_pred, average=average),
        f1=metrics.f1_score(y_true, y_pred, average=average),
        accuracy=metrics.accuracy(y_true, y_pred),
        mae=metrics.mean_absolute_error(y_true, y_pred),
        mse=metrics.mean_squared_error(y_true, y_pred),
        rmse=metrics.root_mean_squared_error(y_true, y_pred),
        pearson=metrics.pearson_correlation(y_true, y_pred),
        spearman=metrics.spearman_correlation(
            [float(y) for y in y_true], [float(y) for y in y_pred]
        ),
        accuracy_within_1=metrics.accuracy_within_k(y_true, y_pred, k=1),
        confusion=metrics.confusion_matrix(y_true, y_pred, labels=RATING_LABELS),
        confidence=confidence,
        cost=cost,
    )
