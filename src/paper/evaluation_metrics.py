"""Metric calculation (precision, recall, F1 and accuracy) for ratings 1-5."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

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


def calculate_paper_metrics(
    papers: Sequence[Evaluated],
    average: Literal["binary", "macro", "micro"] | None = None,
) -> Metrics:
    """Calculate evaluation metrics.

    Args:
        papers: Papers to evaluate. The only requirement is having `y_pred` and `y_true`
            integer properties.
        average: What average mode to use for precision/recall/F1. If None, will use
            'binary' when mode is binary and 'macro' for everything else.

    See also `paper.evaluation_metrics.calculate_metrics`.
    """
    y_pred = [p.y_pred for p in papers]
    y_true = [p.y_true for p in papers]

    return calculate_metrics(y_true, y_pred, average=average)


def calculate_negative_paper_metrics(papers: Sequence[Evaluated]) -> Metrics:
    """Calculate evaluation metrics for the negative label in binary classification.

    This flips the label (true -> false and vice versa), then calculates the metrics.
    """
    y_pred = [p.y_pred for p in papers]
    y_true = [p.y_true for p in papers]

    values = set(y_pred) | set(y_true)
    if values != {0, 1}:
        raise ValueError("Negative paper metrics only make sense for binary results.")

    flipped_y_pred = [1 - y for y in y_pred]
    flipped_y_true = [1 - y for y in y_true]

    return calculate_metrics(flipped_y_true, flipped_y_pred, mode=TargetMode.BIN)


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


def display_regular_negative_macro_metrics(items: Sequence[Evaluated]) -> str:
    """Display regular metrics in addition to negative and macro-averaged."""
    regular = calculate_paper_metrics(items)
    negative = calculate_negative_paper_metrics(items)
    macro = calculate_paper_metrics(items, average="macro")

    out = [
        "> Positive:",
        str(regular),
        "",
        "> Negative:",
        str(negative),
        "",
        "> Macro-averaged:",
        str(macro),
        "",
        f"> {regular.display_confusion()}",
        "",
        "> Table:",
        display_metrics_row(regular, negative, macro),
    ]

    return "\n".join(out)


def display_metrics_row(
    regular: Metrics, negative: Metrics | None = None, macro: Metrics | None = None
) -> str:
    """Build a table with all the given metrics.

    The purpose is to make it easier to copy-paste the results by stacking the outputs.
    """
    header = ["Acc"]
    values = [regular.accuracy]

    for prefix, entry in [("P", regular), ("N", negative), ("M", macro)]:
        if entry:
            header += [f"{prefix}-P", f"{prefix}-R", f"{prefix}-F1"]
            values += [entry.precision, entry.recall, entry.f1]

    values_txt = [f"{x:.4f}" for x in values]

    padding = [max(len(h), len(v)) + 1 for h, v in zip(header, values_txt)]

    header_padded = [f"{h:{pad}}" for h, pad in zip(header, padding)]
    sep_row = ["-" * pad for pad in padding]
    values_padded = [f"{v:{pad}}" for v, pad in zip(values_txt, padding)]

    return "\n".join(
        f"| {' | '.join(row)}|" for row in (header_padded, sep_row, values_padded)
    )


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
    y_true: Sequence[int],
    y_pred: Sequence[int],
    mode: TargetMode | None = None,
    average: Literal["binary", "macro", "micro"] | None = None,
) -> Metrics:
    """Calculate classification metrics for multi-class classification.

    The labels can be either in 1-5 or binary (0/1). This is determined by the range of
    values in the inputs.

    Args:
        y_true: Ground truth labels (values 1-5 or 0/1)
        y_pred: Predicted labels (values 1-5 or 0/1)
        mode: What mode are the labels, either BIN (0/1) or INT (1-5). If absent, we will
            attempt to find the mode by checking the possible values.
        average: What average mode to use for precision/recall/F1. If None, will use
            'binary' when mode is binary and 'macro' for everything else.

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

    if average is None:
        if mode is TargetMode.BIN:
            average = "binary"
        else:
            average = "macro"

    return Metrics(
        precision=metrics.precision(y_true, y_pred, average=average),
        recall=metrics.recall(y_true, y_pred, average=average),
        f1=metrics.f1_score(y_true, y_pred, average=average),
        accuracy=metrics.accuracy(y_true, y_pred),
        mae=metrics.mean_absolute_error(y_true, y_pred),
        mse=metrics.mean_squared_error(y_true, y_pred),
        correlation=metrics.pearson_correlation(y_true, y_pred),
        confusion=metrics.confusion_matrix(y_true, y_pred, labels=mode.labels()),
        mode=mode,
    )
