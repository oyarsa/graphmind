"""Unit tests for the fix_evaluated_rating function."""

import pytest

from paper.gpt.evaluate_paper import GPTFull, fix_evaluated_rating


class MockEval:
    """Mock evaluation result for testing invalid labels.

    Since GPTFull now validates labels on construction (1-5 only), we need a mock
    class to test the clamping behaviour for invalid labels.
    """

    def __init__(self, label: int, rationale: str) -> None:
        self._label = label
        self._rationale = rationale

    @property
    def label(self) -> int:
        """Return the label."""
        return self._label

    @property
    def rationale(self) -> str:
        """Return the rationale."""
        return self._rationale


class TestFixEvaluatedRating:
    """Test cases for fix_evaluated_rating function."""

    @pytest.mark.parametrize(
        ("label", "expected_label"),
        [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ],
    )
    def test_valid_labels_preserved(self, label: int, expected_label: int) -> None:
        """Test that valid labels (1-5) are preserved."""
        rationale = f"Test rationale {label}"
        result = MockEval(label=label, rationale=rationale)
        fixed = fix_evaluated_rating(result)

        assert fixed.label == expected_label
        assert fixed.rationale == rationale
        assert isinstance(fixed, GPTFull)

    @pytest.mark.parametrize(
        ("label", "expected_clamped"),
        [
            # Clamp to 1-5 range
            (0, 1),
            (6, 5),
            (-1, 1),
            (10, 5),
        ],
    )
    def test_invalid_labels_clamped(self, label: int, expected_clamped: int) -> None:
        """Test that invalid labels are clamped to valid range (1-5)."""
        rationale = f"Test rationale {label}"
        result = MockEval(label=label, rationale=rationale)
        fixed = fix_evaluated_rating(result)

        assert fixed.label == expected_clamped
        assert fixed.rationale == rationale

    def test_rationale_preserved_exactly(self) -> None:
        """Test that rationale is preserved exactly, including special characters."""
        special_rationale = "Rationale with\nnewlines\t\tand\t\ttabs and ünicode"
        result = MockEval(label=3, rationale=special_rationale)
        fixed = fix_evaluated_rating(result)
        assert fixed.rationale == special_rationale

    def test_empty_rationale(self) -> None:
        """Test handling of empty rationale."""
        result = MockEval(label=3, rationale="")
        fixed = fix_evaluated_rating(result)
        assert fixed.label == 3
        assert not fixed.rationale

    @pytest.mark.parametrize(
        ("label", "expected_clamped"),
        [
            (-1, 1),
            (-10, 1),
            (-100, 1),
            (-(2**31), 1),
            (1000, 5),
            (999999, 5),
            (2**31 - 1, 5),
        ],
    )
    def test_extreme_invalid_labels(self, label: int, expected_clamped: int) -> None:
        """Test handling of very large and very negative invalid labels."""
        result = MockEval(label=label, rationale="Extreme label")
        fixed = fix_evaluated_rating(result)
        assert fixed.label == expected_clamped
        assert fixed.rationale == "Extreme label"
