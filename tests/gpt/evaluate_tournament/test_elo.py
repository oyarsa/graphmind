"""Tests for Elo rating system in paper.gpt.evaluate_tournament.elo."""

from __future__ import annotations

from math import isclose

from paper.gpt.evaluate_tournament.elo import (
    DEFAULT_ELO,
    K_FACTOR,
    EloPlayer,
    EloTournamentSystem,
    _elo_expected_probabilities,
    _update_elo_rating,
    calculate_elo_rankings,
    calculate_melo_rankings,
)
from paper.gpt.evaluate_tournament.tournament import (
    ComparisonResult,
    MatchResult,
    MatchWinner,
)


def test_elo_player_fresh():
    """Test creating a fresh Elo player."""
    player = EloPlayer.fresh(name="test_player")

    assert player.name == "test_player"
    assert player.rating == DEFAULT_ELO
    assert player.wins == 0
    assert player.losses == 0
    assert player.ties == 0
    assert len(player.match_history) == 0


def test_update_elo_rating():
    """Test Elo rating update function."""
    initial_rating = 1200

    # Win case: actual=1.0, expected=0.5
    new_rating_win = _update_elo_rating(initial_rating, 1.0, 0.5)
    expected_win = initial_rating + K_FACTOR * 0.5
    assert new_rating_win == expected_win

    # Loss case: actual=0.0, expected=0.5
    new_rating_loss = _update_elo_rating(initial_rating, 0.0, 0.5)
    expected_loss = initial_rating - K_FACTOR * 0.5
    assert new_rating_loss == expected_loss

    # Tie case: actual=0.5, expected=0.5
    new_rating_tie = _update_elo_rating(initial_rating, 0.5, 0.5)
    expected_tie = initial_rating  # No change since actual == expected
    assert new_rating_tie == expected_tie


def test_elo_expected_probabilities():
    """Test calculation of expected probabilities."""
    # Same rating: should be 50%/50%
    player_a = EloPlayer.fresh(name="A")
    player_b = EloPlayer.fresh(name="B")
    expected_a, expected_b = _elo_expected_probabilities(player_a, player_b)

    assert isclose(expected_a, 0.5)
    assert isclose(expected_b, 0.5)

    # A has higher rating: A's probability should be higher
    player_a_stronger = EloPlayer(
        name="A", rating=1300, wins=0, losses=0, ties=0, match_history=()
    )

    expected_a, expected_b = _elo_expected_probabilities(player_a_stronger, player_b)
    assert expected_a > 0.5
    assert expected_b < 0.5
    assert isclose(expected_a + expected_b, 1.0)


def test_elo_player_play_match():
    """Test player.play_match functionality."""
    player = EloPlayer.fresh(name="test_player")

    # Test a win
    updated_player = player.play_match(
        opponent="opponent",
        expected_score=0.5,
        actual_score=1.0,
        explanation="Player won",
    )

    assert updated_player.wins == 1
    assert updated_player.losses == 0
    assert updated_player.ties == 0
    assert updated_player.rating > player.rating
    assert len(updated_player.match_history) == 1
    assert updated_player.match_history[0].player == "test_player"
    assert updated_player.match_history[0].opponent == "opponent"
    assert updated_player.match_history[0].score == 1.0

    # Test a loss
    updated_player = player.play_match(
        opponent="opponent",
        expected_score=0.5,
        actual_score=0.0,
        explanation="Player lost",
    )

    assert updated_player.wins == 0
    assert updated_player.losses == 1
    assert updated_player.ties == 0
    assert updated_player.rating < player.rating

    # Test a tie
    updated_player = player.play_match(
        opponent="opponent",
        expected_score=0.5,
        actual_score=0.5,
        explanation="Player tied",
    )

    assert updated_player.wins == 0
    assert updated_player.losses == 0
    assert updated_player.ties == 1
    assert updated_player.rating == player.rating


def test_elo_tournament_system_create(sample_player_names: list[str]):
    """Test creating a new EloTournamentSystem."""
    tournament = EloTournamentSystem.create(sample_player_names, "test_metric")

    assert tournament.metric == "test_metric"
    assert len(tournament.players) == len(sample_player_names)
    assert all(name in tournament.players for name in sample_player_names)
    assert all(player.rating == DEFAULT_ELO for player in tournament.players.values())
    assert len(tournament.matches) == 0


def test_elo_tournament_system_record_match():
    """Test recording a match in EloTournamentSystem."""
    tournament = EloTournamentSystem.create(["A", "B"], "test_metric")

    # A wins against B
    result = MatchResult(winner=MatchWinner.A, explanation="A beats B")
    updated_tournament = tournament.record_match("A", "B", result)

    # Check match was recorded
    assert len(updated_tournament.matches) == 1
    assert updated_tournament.matches[0].player_a == "A"
    assert updated_tournament.matches[0].player_b == "B"
    assert updated_tournament.matches[0].result.winner == MatchWinner.A

    # Check player stats
    assert updated_tournament.players["A"].wins == 1
    assert updated_tournament.players["A"].losses == 0
    assert updated_tournament.players["B"].wins == 0
    assert updated_tournament.players["B"].losses == 1
    assert updated_tournament.players["A"].rating > tournament.players["A"].rating
    assert updated_tournament.players["B"].rating < tournament.players["B"].rating


def test_elo_tournament_system_get_rankings():
    """Test getting rankings from EloTournamentSystem."""
    tournament = EloTournamentSystem.create(["A", "B", "C"], "test_metric")

    # A beats B, B beats C, A beats C
    tournament = tournament.record_match(
        "A", "B", MatchResult(winner=MatchWinner.A, explanation="")
    )
    tournament = tournament.record_match(
        "B", "C", MatchResult(winner=MatchWinner.A, explanation="")
    )
    tournament = tournament.record_match(
        "A", "C", MatchResult(winner=MatchWinner.A, explanation="")
    )

    rankings = tournament.get_rankings()

    # Check order of rankings
    assert len(rankings) == 3
    assert rankings[0].name == "A"  # A should be first (2 wins)
    assert rankings[1].name == "B"  # B should be second (1 win, 1 loss)
    assert rankings[2].name == "C"  # C should be third (2 losses)

    # Check ranks
    assert rankings[0].rank == 1
    assert rankings[1].rank == 2
    assert rankings[2].rank == 3

    # Check win/loss count
    assert rankings[0].wins == 2
    assert rankings[1].wins == 1
    assert rankings[2].wins == 0
    assert rankings[0].losses == 0
    assert rankings[1].losses == 1
    assert rankings[2].losses == 2


def test_calculate_elo_rankings(
    sample_player_names: list[str],
    sample_metrics: list[str],
    sample_comparison_results: list[ComparisonResult],
):
    """Test calculate_elo_rankings function."""
    result = calculate_elo_rankings(
        sample_comparison_results, sample_player_names, sample_metrics
    )

    # Check the structure of the result
    assert result.total_comparisons == len(sample_comparison_results)
    assert set(result.tournaments.keys()) == set(sample_metrics)

    # Check that we have rankings for each player
    for name in sample_player_names:
        assert name in result.overall_ranks

    # Basic check of consistency
    for metric in sample_metrics:
        rankings = result.tournaments[metric].get_rankings()
        assert len(rankings) == len(sample_player_names)

        # Check that ranks are unique and complete
        ranks = [r.rank for r in rankings]
        assert sorted(ranks) == list(range(1, len(sample_player_names) + 1))


def test_calculate_melo_rankings(
    sample_player_names: list[str],
    sample_metrics: list[str],
    sample_comparison_results: list[ComparisonResult],
):
    """Test calculate_melo_rankings function."""
    result = calculate_melo_rankings(
        sample_comparison_results, sample_player_names, sample_metrics, 3, 42
    )

    # Check the structure of the result
    assert result.total_comparisons == len(sample_comparison_results)
    assert set(result.tournaments.keys()) == set(sample_metrics)

    # Check that we have rankings for each player
    for name in sample_player_names:
        assert name in result.overall_ranks

    # Basic check of consistency
    for metric in sample_metrics:
        rankings = result.tournaments[metric].get_rankings()
        assert len(rankings) == len(sample_player_names)

        # Check that ranks are unique and complete
        ranks = [r.rank for r in rankings]
        assert sorted(ranks) == list(range(1, len(sample_player_names) + 1))
