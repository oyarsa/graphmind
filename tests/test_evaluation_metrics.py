import pytest
from paper.evaluation_metrics import TargetMode, _guess_target_mode


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
