"""Type-safe wrappers around scipy and sklearn metrics."""
# pyright: basic

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import stats
from sklearn import metrics as skmetrics


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Calculate Pearson correlation coefficient between two sequences.

    Returns None if correlation is undefined (e.g., if one sequence has zero variance).
    """
    if len(set(y)) == 1 or len(set(x)) == 1:  # Either sequence has zero variance
        return None

    return float(stats.pearsonr(x, y).correlation)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Calculate Spearman correlation coefficient between two sequences.

    Returns None if correlation is undefined (e.g., if one sequence has zero variance).
    """
    if len(set(y)) == 1 or len(set(x)) == 1:
        return None
    return float(stats.spearmanr(x, y).statistic)  # type: ignore


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate mean absolute error between true and predicted values."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate mean squared error between true and predicted values."""
    return float(np.mean(np.square(np.array(y_true) - np.array(y_pred))))


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Calculate classification accuracy score."""
    return float(skmetrics.accuracy_score(y_true, y_pred))


def precision(
    y_true: Sequence[int], y_pred: Sequence[int], average: str = "macro"
) -> float:
    """Calculate precision score with specified averaging method."""
    return float(
        skmetrics.precision_score(y_true, y_pred, average=average, zero_division=0)  # type: ignore
    )


def recall(
    y_true: Sequence[int], y_pred: Sequence[int], average: str = "macro"
) -> float:
    """Calculate recall score with specified averaging method."""
    return float(
        skmetrics.recall_score(y_true, y_pred, average=average, zero_division=0)  # type: ignore
    )


def f1_score(
    y_true: Sequence[int], y_pred: Sequence[int], average: str = "macro"
) -> float:
    """Calculate F1 score with specified averaging method."""
    return float(skmetrics.f1_score(y_true, y_pred, average=average, zero_division=0))  # type: ignore


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Iterable[int] | None = None
) -> list[list[int]]:
    """Calculate confusion matrix with optional label specification."""
    if labels is not None:
        labels = list(labels)
    return skmetrics.confusion_matrix(y_true, y_pred, labels=labels).tolist()


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate root mean squared error between true and predicted values."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def accuracy_within_k(
    y_true: Sequence[int], y_pred: Sequence[int], k: int = 1
) -> float:
    """Calculate accuracy allowing k points of error.

    For ordinal ratings (1-5), this counts predictions as correct if they are
    within k points of the true value.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        k: Maximum allowed error (default 1).

    Returns:
        Proportion of predictions within k of the true value.
    """
    if len(y_true) == 0:
        return 0.0
    correct = sum(abs(true - pred) <= k for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)
