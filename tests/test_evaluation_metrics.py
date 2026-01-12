"""Test classification evaluation metrics for 1-5 integer ratings."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose

import pytest

from paper.evaluation_metrics import TargetMode, _guess_target_mode, calculate_metrics
from paper.types import Immutable


def test_guess_target_mode_always_returns_int() -> None:
    """Test that _guess_target_mode always returns INT mode."""
    # Regardless of input values, mode should be INT
    assert _guess_target_mode([1, 2], [3, 4]) == TargetMode.INT
    assert _guess_target_mode([1], [5]) == TargetMode.INT
    assert _guess_target_mode([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == TargetMode.INT


@dataclass(frozen=True)
class ExpectedMetrics:
    """Expected metrics for classification tests."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    mae: float
    rmse: float


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            ExpectedMetrics(
                precision=1.0, recall=1.0, f1=1.0, accuracy=1.0, mae=0.0, rmse=0.0
            ),
            id="Perfect prediction",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            ExpectedMetrics(
                precision=0.2, recall=0.2, f1=0.2, accuracy=0.2, mae=2.4, rmse=2.83
            ),
            id="Reversed predictions",
        ),
        pytest.param(
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            ExpectedMetrics(
                precision=1.0, recall=1.0, f1=1.0, accuracy=1.0, mae=0.0, rmse=0.0
            ),
            id="All same class",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 5],
            ExpectedMetrics(
                precision=0.1, recall=0.2, f1=0.13, accuracy=0.2, mae=0.8, rmse=0.894
            ),
            id="Off by one (mostly)",
        ),
    ],
)
def test_calculate_metrics_int(
    y_true: list[int],
    y_pred: list[int],
    expected: ExpectedMetrics,
) -> None:
    """Test calculate_metrics function for integer rating classification."""
    result = calculate_metrics(y_true, y_pred, mode=TargetMode.INT)

    assert isclose(result.precision, expected.precision, abs_tol=1e-2), "Precision"
    assert isclose(result.recall, expected.recall, abs_tol=1e-2), "Recall"
    assert isclose(result.f1, expected.f1, abs_tol=1e-2), "F1"
    assert isclose(result.accuracy, expected.accuracy, abs_tol=1e-2), "Accuracy"
    assert isclose(result.mae, expected.mae, abs_tol=1e-2), "MAE"
    assert isclose(result.rmse, expected.rmse, abs_tol=1e-2), "RMSE"


@pytest.mark.parametrize(
    ("y_true", "y_pred", "error_msg"),
    [
        # Different lengths
        ([1, 2, 3, 4], [1, 2, 3], "Input sequences must have the same length"),
        # Empty sequences
        ([], [], "Input sequences cannot be empty"),
    ],
)
def test_calculate_metrics_errors(
    y_true: list[int], y_pred: list[int], error_msg: str
) -> None:
    """Test that calculate_metrics raises appropriate errors for invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        calculate_metrics(y_true, y_pred, mode=TargetMode.INT)


class EvaluateResult(Immutable):
    """Mock evaluation result for testing."""

    y_true: int
    y_pred: int


def y_sequences_to_result(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> list[EvaluateResult]:
    """Convert sequences to list of EvaluateResult objects."""
    return [EvaluateResult(y_pred=p, y_true=t) for p, t in zip(y_pred, y_true)]


def test_accuracy_within_1() -> None:
    """Test accuracy within ±1 metric."""
    # Perfect predictions
    result = calculate_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], mode=TargetMode.INT)
    assert result.accuracy_within_1 == 1.0

    # Off by one is still correct
    result = calculate_metrics([1, 2, 3, 4, 5], [2, 3, 4, 5, 4], mode=TargetMode.INT)
    assert result.accuracy_within_1 == 1.0

    # Some off by more than one
    result = calculate_metrics([1, 1, 5, 5], [3, 1, 3, 5], mode=TargetMode.INT)
    assert result.accuracy_within_1 == 0.5


def test_correlation_metrics() -> None:
    """Test Pearson and Spearman correlation metrics."""
    # Perfect positive correlation
    result = calculate_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], mode=TargetMode.INT)
    assert result.pearson is not None
    assert isclose(result.pearson, 1.0, abs_tol=1e-6)
    assert result.spearman is not None
    assert isclose(result.spearman, 1.0, abs_tol=1e-6)

    # Perfect negative correlation
    result = calculate_metrics([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], mode=TargetMode.INT)
    assert result.pearson is not None
    assert isclose(result.pearson, -1.0, abs_tol=1e-6)
    assert result.spearman is not None
    assert isclose(result.spearman, -1.0, abs_tol=1e-6)

    # No correlation possible (constant values)
    result = calculate_metrics([3, 3, 3, 3], [3, 3, 3, 3], mode=TargetMode.INT)
    assert result.pearson is None
    assert result.spearman is None


def test_confusion_matrix_shape() -> None:
    """Test that confusion matrix has correct shape for INT mode."""
    result = calculate_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], mode=TargetMode.INT)
    assert len(result.confusion) == 5  # 5 classes for 1-5 ratings
    assert all(len(row) == 5 for row in result.confusion)
