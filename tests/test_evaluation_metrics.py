from dataclasses import dataclass
from math import isclose

import pytest

from paper.evaluation_metrics import TargetMode, _guess_target_mode, calculate_metrics


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 1, 0], [0, 0, 1], TargetMode.BIN),
        ([0], [1], TargetMode.BIN),
        ([0], [2], TargetMode.INT),
        ([0, 1, 2], [3, 0, 4], TargetMode.INT),
        ([0, 0, 0], [0, 0, 0], TargetMode.INT),  # No way to know.
    ],
)
def test_guess_target_mode(
    y_pred: list[int], y_true: list[int], expected: TargetMode
) -> None:
    assert _guess_target_mode(y_pred, y_true) == expected


@dataclass(frozen=True)
class ExpectedMetrics:
    """Expected metrics for classification tests."""

    precision: float
    recall: float
    f1: float
    accuracy: float


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        pytest.param(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            ExpectedMetrics(precision=1.0, recall=1.0, f1=1.0, accuracy=1.0),
            id="Perfect prediction",
        ),
        pytest.param(
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            ExpectedMetrics(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0),
            id="All incorrect",
        ),
        pytest.param(
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
            ExpectedMetrics(precision=0.5, recall=0.5, f1=0.5, accuracy=0.5),
            id="Mixed case with balanced classes",
        ),
        pytest.param(
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
            ExpectedMetrics(precision=0.5, recall=0.5, f1=0.5, accuracy=0.667),
            id="Imbalanced case",
        ),
        pytest.param(
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            ExpectedMetrics(precision=0, recall=0, f1=0, accuracy=0.5),
            id="All predicted as one class (0)",
        ),
        pytest.param(
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            ExpectedMetrics(precision=0.5, recall=1, f1=0.666, accuracy=0.5),
            id="All predicted as one class (1)",
        ),
    ],
)
def test_calculate_metrics_binary(
    y_true: list[int], y_pred: list[int], expected: ExpectedMetrics
):
    """Test calculate_metrics function for binary classification."""
    result = calculate_metrics(y_true, y_pred, mode=TargetMode.BIN)

    assert isclose(result.precision, expected.precision, abs_tol=1e-3), "Precision"
    assert isclose(result.recall, expected.recall, abs_tol=1e-3), "Recall"
    assert isclose(result.f1, expected.f1, abs_tol=1e-3), "F1"
    assert isclose(result.accuracy, expected.accuracy, abs_tol=1e-3), "Accuracy"


@pytest.mark.parametrize(
    "y_true, y_pred, error_msg",
    [
        # Different lengths
        ([0, 1, 0, 1], [0, 1, 0], "Input sequences must have the same length"),
        # Empty sequences
        ([], [], "Input sequences cannot be empty"),
    ],
)
def test_calculate_metrics_errors(y_true: list[int], y_pred: list[int], error_msg: str):
    """Test that calculate_metrics raises appropriate errors for invalid inputs."""
    with pytest.raises(ValueError, match=error_msg):
        calculate_metrics(y_true, y_pred, mode=TargetMode.BIN)
