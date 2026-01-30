"""Tests for Semantic Scholar utility functions."""

import pytest

from paper.semantic_scholar.info import is_valid_title_match


class TestIsValidTitleMatch:
    """Tests for is_valid_title_match function."""

    @pytest.mark.parametrize(
        ("searched", "returned", "expected"),
        [
            # Exact match
            ("Attention Is All You Need", "Attention Is All You Need", True),
            # Case insensitive match
            ("attention is all you need", "Attention Is All You Need", True),
            # Minor variations (should still match)
            ("Attention Is All You Need", "Attention is All you Need", True),
            # Slightly different but high similarity
            (
                "Layer Normalization",
                "Layer normalization",
                True,
            ),
            # Completely different papers - the actual bug case
            (
                "arXiv preprint arXiv:1511.06114",
                "ON SOME DETERMINANTS INVOLVING THE TANGENT FUNCTION",
                False,
            ),
            # Journal name vs paper title
            (
                "Journal of Machine Learning Research",
                "A Brief Survey of Machine Learning and Deep Learning Techniques",
                False,
            ),
            # Short generic term matching long specific title
            (
                "CoRR",
                "Spontaneous Reconstruction of Copper Active Sites",
                False,
            ),
            # Similar topic but different paper
            (
                "Neural Machine Translation",
                "Deep Learning for Natural Language Processing",
                False,
            ),
            # Substring match - partial ratio gives 100% for substrings
            # This is expected behaviour of fuzzy_partial_ratio
            (
                "Attention",
                "Self-Attention with Relative Position Representations",
                True,  # Partial ratio matches substrings
            ),
            # Empty strings
            ("", "", True),  # Both empty is technically a match
            ("Some Title", "", False),
            ("", "Some Title", False),
        ],
        ids=[
            "exact_match",
            "case_insensitive",
            "minor_variations",
            "normalisation_match",
            "arxiv_vs_tangent_paper",
            "journal_vs_survey_paper",
            "corr_vs_chemistry_paper",
            "similar_topic_different_paper",
            "substring_not_same_paper",
            "both_empty",
            "searched_empty",
            "returned_empty",
        ],
    )
    def test_title_match_validation(
        self, searched: str, returned: str, expected: bool
    ) -> None:
        """Test title match validation with various inputs."""
        result = is_valid_title_match(searched, returned)
        assert result == expected

    @pytest.mark.parametrize(
        ("searched", "returned", "min_ratio", "expected"),
        [
            # With custom threshold - stricter (95 ratio matches 95 exactly)
            ("Layer Normalization", "Layer Normalisation", 96, False),
            # With custom threshold - more lenient
            ("Neural Networks", "Deep Neural Network Architectures", 50, True),
            # Threshold at 0 - everything matches
            ("Completely Different", "Unrelated Text", 0, True),
            # Threshold at 100 - only exact matches
            ("Exact Title", "Exact Title", 100, True),
            # Case-insensitive: title_ratio uses casefold, so case doesn't matter
            ("Exact Title", "exact title", 100, True),
        ],
        ids=[
            "strict_threshold_rejects",
            "lenient_threshold_accepts",
            "zero_threshold",
            "perfect_threshold_exact",
            "case_insensitive_at_100",
        ],
    )
    def test_custom_threshold(
        self, searched: str, returned: str, min_ratio: int, expected: bool
    ) -> None:
        """Test title match validation with custom thresholds."""
        result = is_valid_title_match(searched, returned, min_ratio=min_ratio)
        assert result == expected
