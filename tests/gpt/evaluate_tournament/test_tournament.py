"""Tests for tournament functionality in paper.gpt.evaluate_tournament.tournament."""

from __future__ import annotations
from math import isclose

import pytest
from collections.abc import Collection

from paper.gpt.evaluate_tournament.tournament import (
    MatchResult,
    MatchWinner,
    PlayerRank,
    TournamentSystem,
    create_tournament_result,
    ComparisonResult,
    display_head_to_head,
)


class MockTournamentSystem(TournamentSystem):
    """Mock tournament system for testing."""

    metric: str
    players: dict[str, dict[str, int]]
    ratings: dict[str, float]

    @classmethod
    def create(cls, item_names: Collection[str], metric: str) -> MockTournamentSystem:
        """Create a new tournament system."""
        return cls(
            metric=metric,
            players={name: {"wins": 0, "losses": 0, "ties": 0} for name in item_names},
            ratings={name: 1200.0 for name in item_names},
        )

    def record_match(
        self,
        player_a_name: str,
        player_b_name: str,
        result: MatchResult,
    ) -> MockTournamentSystem:
        """Record a match result."""
        new_players = self.players.copy()
        new_ratings = self.ratings.copy()

        # Update player stats based on match result
        match result.winner:
            case MatchWinner.A:
                new_players[player_a_name]["wins"] += 1
                new_players[player_b_name]["losses"] += 1
                new_ratings[player_a_name] += 10
                new_ratings[player_b_name] -= 10
            case MatchWinner.B:
                new_players[player_a_name]["losses"] += 1
                new_players[player_b_name]["wins"] += 1
                new_ratings[player_a_name] -= 10
                new_ratings[player_b_name] += 10
            case MatchWinner.TIE:
                new_players[player_a_name]["ties"] += 1
                new_players[player_b_name]["ties"] += 1

        return MockTournamentSystem(
            metric=self.metric,
            players=new_players,
            ratings=new_ratings,
        )

    def get_rankings(self) -> list[PlayerRank]:
        """Get rankings for all players."""
        player_ratings = [
            (name, self.ratings[name], self.players[name]) for name in self.players
        ]
        sorted_players = sorted(player_ratings, key=lambda x: x[1], reverse=True)

        return [
            PlayerRank(
                rank=i,
                name=name,
                rating=rating,
                wins=stats["wins"],
                losses=stats["losses"],
                ties=stats["ties"],
            )
            for i, (name, rating, stats) in enumerate(sorted_players, 1)
        ]


@pytest.fixture
def sample_tournaments(
    sample_player_names: list[str], sample_metrics: list[str]
) -> dict[str, MockTournamentSystem]:
    """Sample tournaments for testing."""
    tournaments: dict[str, MockTournamentSystem] = {}

    for metric in sample_metrics:
        tournament = MockTournamentSystem.create(sample_player_names, metric)

        # Record some matches to create different rankings per metric
        if metric == "clarity":
            # model_a beats model_b
            tournament = tournament.record_match(
                "model_a", "model_b", MatchResult(winner=MatchWinner.A, explanation="")
            )
            # model_a beats model_c
            tournament = tournament.record_match(
                "model_a", "model_c", MatchResult(winner=MatchWinner.A, explanation="")
            )
            # model_b beats model_c
            tournament = tournament.record_match(
                "model_b", "model_c", MatchResult(winner=MatchWinner.A, explanation="")
            )
        else:  # factuality
            # model_b beats model_a
            tournament = tournament.record_match(
                "model_a", "model_b", MatchResult(winner=MatchWinner.B, explanation="")
            )
            # model_c beats model_a
            tournament = tournament.record_match(
                "model_a", "model_c", MatchResult(winner=MatchWinner.B, explanation="")
            )
            # model_b beats model_c
            tournament = tournament.record_match(
                "model_b", "model_c", MatchResult(winner=MatchWinner.A, explanation="")
            )

        tournaments[metric] = tournament

    return tournaments


def test_tournament_record_match(sample_player_names: list[str]):
    """Test the TournamentSystem record_match method using our mock implementation."""
    # Create a tournament
    tournament = MockTournamentSystem.create(sample_player_names, "clarity")

    # Record a match
    result = MatchResult(winner=MatchWinner.A, explanation="A wins")
    updated_tournament = tournament.record_match("model_a", "model_b", result)

    # Check that the tournament was updated correctly
    assert updated_tournament.players["model_a"]["wins"] == 1
    assert updated_tournament.players["model_b"]["losses"] == 1

    # Create a tournament for a different metric
    tournament2 = MockTournamentSystem.create(sample_player_names, "factuality")

    # Check that it has no wins/losses initially
    assert tournament2.players["model_a"]["wins"] == 0
    assert tournament2.players["model_b"]["losses"] == 0


def test_tournament_result_overall_ranks(
    sample_tournaments: dict[str, MockTournamentSystem],
    sample_metrics: list[str],
    sample_comparison_results: list[ComparisonResult],
):
    """Test overall rankings calculation through TournamentResult."""
    result = create_tournament_result(
        comparison_results=sample_comparison_results,
        metrics=sample_metrics,
        tournaments=sample_tournaments,
    )

    overall_ranks = result.overall_ranks

    # Check results for each model
    assert "model_a" in overall_ranks
    assert "model_b" in overall_ranks
    assert "model_c" in overall_ranks

    # model_a is 1st in clarity, 3rd in factuality
    model_a_stats = overall_ranks["model_a"]
    assert model_a_stats.ranks == [1, 3]
    assert isclose(model_a_stats.mean_rank, 2.0)
    assert model_a_stats.best_rank == 1
    assert model_a_stats.worst_rank == 3
    assert model_a_stats.metric_ranks["clarity"] == 1
    assert model_a_stats.metric_ranks["factuality"] == 3

    # model_b is 2nd in clarity, 1st in factuality
    model_b_stats = overall_ranks["model_b"]
    assert model_b_stats.ranks == [2, 1]
    assert isclose(model_b_stats.mean_rank, 1.5)
    assert model_b_stats.best_rank == 1
    assert model_b_stats.worst_rank == 2
    assert model_b_stats.metric_ranks["clarity"] == 2
    assert model_b_stats.metric_ranks["factuality"] == 1


def test_create_tournament_result(
    sample_comparison_results: list[ComparisonResult],
    sample_tournaments: dict[str, MockTournamentSystem],
    sample_metrics: list[str],
):
    """Test create_tournament_result function."""
    result = create_tournament_result(
        comparison_results=sample_comparison_results,
        metrics=sample_metrics,
        tournaments=sample_tournaments,
    )

    # Check the result structure
    assert result.total_comparisons == len(sample_comparison_results)
    assert set(result.tournaments.keys()) == set(sample_metrics)

    # Check overall ranks
    assert "model_a" in result.overall_ranks
    assert "model_b" in result.overall_ranks
    assert "model_c" in result.overall_ranks


def test_display_head_to_head(
    sample_comparison_results: list[ComparisonResult],
    sample_player_names: list[str],
    sample_metrics: list[str],
):
    """Test display_head_to_head function."""
    result = display_head_to_head(
        sample_comparison_results, sample_player_names, sample_metrics
    )

    # Check that the result is a non-empty string
    assert len(result) > 0

    # Check that result contains metric names
    for metric in sample_metrics:
        assert metric.capitalize() in result

    # Check that all players are mentioned
    for player in sample_player_names:
        assert player in result
