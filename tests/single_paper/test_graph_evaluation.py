"""Tests for graph_evaluation module."""

import pytest

from paper.single_paper.graph_evaluation import calculate_evidence_distribution


@pytest.mark.parametrize(
    ("total_semantic", "total_citations", "expected"),
    [
        # Basic cases - preferred distribution (3 semantic + 2 citations)
        (3, 2, (3, 2)),
        (5, 5, (3, 2)),
        (10, 10, (3, 2)),
        # Not enough semantic - fill with citations
        (0, 5, (0, 5)),
        (1, 5, (1, 4)),
        (2, 5, (2, 3)),
        # Not enough citations - fill with semantic
        (5, 0, (5, 0)),
        (5, 1, (4, 1)),
        (4, 1, (4, 1)),
        # Both sources limited
        (0, 0, (0, 0)),
        (1, 1, (1, 1)),
        (2, 2, (2, 2)),
        (3, 1, (3, 1)),
        (1, 3, (1, 3)),
        # Edge cases - exactly at boundaries
        (3, 0, (3, 0)),
        (0, 2, (0, 2)),
        (3, 3, (3, 2)),
        # Tie-breaker: when remaining counts are equal, prefer citations
        (4, 3, (3, 2)),  # remaining_sem=1, remaining_cit=1, cit wins tie
    ],
)
def test_calculate_evidence_distribution(
    total_semantic: int, total_citations: int, expected: tuple[int, int]
) -> None:
    """Test evidence distribution calculation."""
    result = calculate_evidence_distribution(total_semantic, total_citations)
    assert result == expected
