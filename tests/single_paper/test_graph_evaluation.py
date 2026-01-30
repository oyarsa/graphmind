"""Tests for graph_evaluation module."""

import pytest

from paper.single_paper.graph_evaluation import (
    calculate_evidence_distribution,
    format_author_names,
)


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


@pytest.mark.parametrize(
    ("names", "max_display", "expected"),
    [
        # Empty list returns default
        ([], 2, "Unknown authors"),
        # Single author
        (["Alice"], 2, "Alice"),
        # Two authors (at max_display=2)
        (["Alice", "Bob"], 2, "Alice, Bob"),
        # Three authors with max_display=2 uses "et al."
        (["Alice", "Bob", "Charlie"], 2, "Alice et al."),
        # Many authors
        (["A", "B", "C", "D", "E"], 2, "A et al."),
        # Custom max_display of 3
        (["Alice", "Bob", "Charlie"], 3, "Alice, Bob, Charlie"),
        (["A", "B", "C", "D"], 3, "A et al."),
        # Custom max_display of 1
        (["Alice", "Bob"], 1, "Alice et al."),
        (["Alice"], 1, "Alice"),
    ],
)
def test_format_author_names(names: list[str], max_display: int, expected: str) -> None:
    """Test author name formatting."""
    result = format_author_names(names, max_display=max_display)
    assert result == expected


def test_format_author_names_custom_default() -> None:
    """Test format_author_names with custom default value."""
    result = format_author_names([], default="N/A")
    assert result == "N/A"
