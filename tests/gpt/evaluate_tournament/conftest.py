"""Shared fixtures for tournament tests."""

from __future__ import annotations

import pytest

from paper.gpt.evaluate_tournament.tournament import (
    MatchResult,
    MatchWinner,
    ComparisonResult,
)
from paper import peerread as pr


@pytest.fixture
def sample_player_names() -> list[str]:
    """Sample player names for testing."""
    return ["model_a", "model_b", "model_c"]


@pytest.fixture
def sample_metrics() -> list[str]:
    """Sample metrics for testing."""
    return ["clarity", "factuality"]


@pytest.fixture
def sample_paper() -> pr.Paper:
    """Create a sample paper core for testing."""
    return pr.Paper(
        title="Sample Paper",
        authors=["Jane Doe"],
        abstract="Abstract text",
        approval=None,
        conference="Sample Conference",
        year=2023,
        sections=[],
        references=[],
        reviews=[
            pr.PaperReview(
                rating=5,
                confidence=None,
                rationale="",
            )
        ],
    )


@pytest.fixture
def sample_comparison_results(
    sample_paper: pr.Paper, sample_player_names: list[str], sample_metrics: list[str]
) -> list[ComparisonResult]:
    """Create sample comparison results for testing."""
    results: list[ComparisonResult] = []

    # Create all pairwise matchups for each metric
    for metric in sample_metrics:
        for i, player_a in enumerate(sample_player_names):
            for player_b in sample_player_names[i + 1 :]:
                # Alternating winners based on players and metrics
                if (player_a == "model_a" and metric == "clarity") or (
                    player_b == "model_b" and metric == "factuality"
                ):
                    winner = MatchWinner.A
                else:
                    winner = MatchWinner.B

                results.append(
                    ComparisonResult(
                        paper=sample_paper,
                        item_a=player_a,
                        item_b=player_b,
                        rationale_a="Rationale for A",
                        rationale_b="Rationale for B",
                        metric=metric,
                        result=MatchResult(winner=winner, explanation="Explanation"),
                    )
                )

    return results
