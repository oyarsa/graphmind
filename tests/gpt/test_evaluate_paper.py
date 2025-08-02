"""Unit tests for the fix_evaluated_rating function."""

import pytest

from paper.evaluation_metrics import TargetMode
from paper.gpt.evaluate_paper import GPTFull, fix_evaluated_rating


class TestFixEvaluatedRating:
    """Test cases for fix_evaluated_rating function."""

    @pytest.mark.parametrize(
        ("label", "target_mode", "expected_label"),
        [
            # Binary mode valid labels
            (0, TargetMode.BIN, 0),
            (1, TargetMode.BIN, 1),
            # Uncertain mode valid labels
            (0, TargetMode.UNCERTAIN, 0),
            (1, TargetMode.UNCERTAIN, 1),
            (2, TargetMode.UNCERTAIN, 2),
            # Int mode valid labels
            (1, TargetMode.INT, 1),
            (2, TargetMode.INT, 2),
            (3, TargetMode.INT, 3),
            (4, TargetMode.INT, 4),
            (5, TargetMode.INT, 5),
        ],
    )
    def test_valid_labels_preserved(
        self, label: int, target_mode: TargetMode, expected_label: int
    ) -> None:
        """Test that valid labels are preserved for each target mode."""
        rationale = f"Test rationale {label}"
        result = GPTFull(label=label, rationale=rationale)
        fixed = fix_evaluated_rating(result, target_mode)

        assert fixed.label == expected_label
        assert fixed.rationale == rationale
        assert isinstance(fixed, GPTFull)

    @pytest.mark.parametrize(
        ("label", "target_mode"),
        [
            # Binary mode invalid labels
            (2, TargetMode.BIN),
            (3, TargetMode.BIN),
            (-1, TargetMode.BIN),
            (10, TargetMode.BIN),
            # Uncertain mode invalid labels
            (3, TargetMode.UNCERTAIN),
            (4, TargetMode.UNCERTAIN),
            (-1, TargetMode.UNCERTAIN),
            (10, TargetMode.UNCERTAIN),
            # Int mode invalid labels
            (0, TargetMode.INT),
            (6, TargetMode.INT),
            (-1, TargetMode.INT),
            (10, TargetMode.INT),
        ],
    )
    def test_invalid_labels_converted_to_zero(
        self, label: int, target_mode: TargetMode
    ) -> None:
        """Test that invalid labels are converted to 0."""
        rationale = f"Test rationale {label}"
        result = GPTFull(label=label, rationale=rationale)
        fixed = fix_evaluated_rating(result, target_mode)

        assert fixed.label == 0
        assert fixed.rationale == rationale

    def test_default_target_mode_is_binary(self) -> None:
        """Test that default target mode is BIN."""
        # Valid for binary
        result = GPTFull(label=1, rationale="Valid binary")
        fixed = fix_evaluated_rating(result)
        assert fixed.label == 1
        assert fixed.rationale == "Valid binary"

        # Invalid for binary (should convert to 0)
        result = GPTFull(label=3, rationale="Invalid binary")
        fixed = fix_evaluated_rating(result)
        assert fixed.label == 0
        assert fixed.rationale == "Invalid binary"

    def test_rationale_preserved_exactly(self) -> None:
        """Test that rationale is preserved exactly, including special characters."""
        special_rationale = "Rationale with\nnewlines\t\tand\t\ttabs and ünicode"
        result = GPTFull(label=1, rationale=special_rationale)
        fixed = fix_evaluated_rating(result, TargetMode.BIN)
        assert fixed.rationale == special_rationale

    def test_empty_rationale(self) -> None:
        """Test handling of empty rationale."""
        result = GPTFull(label=1, rationale="")
        fixed = fix_evaluated_rating(result, TargetMode.BIN)
        assert fixed.label == 1
        assert not fixed.rationale

    @pytest.mark.parametrize(
        "label",
        [-1, -10, -100, -(2**31), 1000, 999999, 2**31 - 1],
    )
    def test_extreme_invalid_labels(self, label: int) -> None:
        """Test handling of very large and very negative invalid labels."""
        result = GPTFull(label=label, rationale="Extreme label")
        fixed = fix_evaluated_rating(result, TargetMode.BIN)
        assert fixed.label == 0
        assert fixed.rationale == "Extreme label"
