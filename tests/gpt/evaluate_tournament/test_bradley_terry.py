"""Tests for Bradley-Terry rating system in paper.gpt.evaluate_tournament.bradley_terry."""

from __future__ import annotations
import itertools


from paper.gpt.evaluate_tournament.bradley_terry import (
    BradleyTerryPlayer,
    BradleyTerryTournamentSystem,
    _bt_update_strengths,
    calculate_bradley_terry_rankings,
    DEFAULT_BT_STRENGTH,
)
from paper.gpt.evaluate_tournament.tournament import (
    MatchResult,
    MatchWinner,
    ComparisonResult,
)


def test_bradley_terry_player_fresh():
    """Test creating a fresh Bradley-Terry player."""
    player = BradleyTerryPlayer.fresh(name="test_player")

    assert player.name == "test_player"
    assert player.strength == DEFAULT_BT_STRENGTH
    assert player.wins == 0
    assert player.losses == 0
    assert player.ties == 0
    assert len(player.match_history) == 0


def test_bradley_terry_player_play_match():
    """Test player.play_match functionality."""
    player = BradleyTerryPlayer.fresh(name="test_player")

    # Test a win
    updated_player = player.play_match(
        opponent="opponent", actual_score=1.0, explanation="Player won"
    )

    assert updated_player.wins == 1
    assert updated_player.losses == 0
    assert updated_player.ties == 0
    assert (
        updated_player.strength == player.strength
    )  # Strength is unchanged until global update
    assert len(updated_player.match_history) == 1
    assert updated_player.match_history[0].player == "test_player"
    assert updated_player.match_history[0].opponent == "opponent"
    assert updated_player.match_history[0].score == 1.0

    # Test a loss
    updated_player = player.play_match(
        opponent="opponent", actual_score=0.0, explanation="Player lost"
    )

    assert updated_player.wins == 0
    assert updated_player.losses == 1
    assert updated_player.ties == 0
    assert (
        updated_player.strength == player.strength
    )  # Strength is unchanged until global update

    # Test a tie
    updated_player = player.play_match(
        opponent="opponent", actual_score=0.5, explanation="Player tied"
    )

    assert updated_player.wins == 0
    assert updated_player.losses == 0
    assert updated_player.ties == 1
    assert (
        updated_player.strength == player.strength
    )  # Strength is unchanged until global update


def test_bradley_terry_player_set_strength():
    """Test player.set_strength functionality."""
    player = BradleyTerryPlayer.fresh(name="test_player")

    # Add some match history
    player = player.play_match("opponent", 1.0, "Win")
    player = player.play_match("opponent2", 0.0, "Loss")

    # Update strength
    new_strength = 1.5
    updated_player = player.set_strength(new_strength)

    # Verify strength is updated but other fields are preserved
    assert updated_player.strength == new_strength
    assert updated_player.name == player.name
    assert updated_player.wins == player.wins
    assert updated_player.losses == player.losses
    assert updated_player.ties == player.ties
    assert updated_player.match_history == player.match_history


def test_bt_update_strengths():
    """Test _bt_update_strengths function."""
    # Create players with known strengths
    players = {
        "A": BradleyTerryPlayer.fresh(name="A"),
        "B": BradleyTerryPlayer.fresh(name="B"),
        "C": BradleyTerryPlayer.fresh(name="C"),
    }

    # Define win counts and match counts
    # A beats B twice, B beats C twice, C beats A twice
    win_counts = {
        ("A", "B"): 2.0,
        ("B", "A"): 0.0,
        ("B", "C"): 2.0,
        ("C", "B"): 0.0,
        ("C", "A"): 2.0,
        ("A", "C"): 0.0,
    }

    match_counts = {
        ("A", "B"): 2.0,
        ("B", "C"): 2.0,
        ("C", "B"): 2.0,
        # Symmetric
        ("B", "A"): 2.0,
        ("C", "A"): 2.0,
        ("A", "C"): 2.0,
    }

    # Update strengths
    updated_players = _bt_update_strengths(players, win_counts, match_counts)

    # Verify each player was updated and strengths make sense
    assert all(name in updated_players for name in players)

    # In a rock-paper-scissors scenario, all players should have similar strengths
    strengths = [player.strength for player in updated_players.values()]
    max_strength = max(strengths)
    min_strength = min(strengths)

    # The ratio between max and min should be reasonable
    # (not too extreme since we have a circular dominance pattern)
    assert max_strength / min_strength < 5.0


def test_bradley_terry_tournament_system_create(sample_player_names: list[str]):
    """Test creating a new BradleyTerryTournamentSystem."""
    tournament = BradleyTerryTournamentSystem.create(sample_player_names, "test_metric")

    assert tournament.metric == "test_metric"
    assert len(tournament.players) == len(sample_player_names)
    assert all(name in tournament.players for name in sample_player_names)
    assert all(
        player.strength == DEFAULT_BT_STRENGTH for player in tournament.players.values()
    )
    assert len(tournament.matches) == 0


def test_bradley_terry_tournament_record_match():
    """Test recording a match in BradleyTerryTournamentSystem."""
    tournament = BradleyTerryTournamentSystem.create(["A", "B"], "test_metric")

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

    # After a single match, the winner should have higher strength than the loser
    assert (
        updated_tournament.players["A"].strength
        > updated_tournament.players["B"].strength
    )


def test_bradley_terry_tournament_get_rankings():
    """Test getting rankings from BradleyTerryTournamentSystem."""
    tournament = BradleyTerryTournamentSystem.create(["A", "B", "C"], "test_metric")

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


def test_calculate_bradley_terry_rankings(
    sample_player_names: list[str],
    sample_metrics: list[str],
    sample_comparison_results: list[ComparisonResult],
):
    """Test calculate_bradley_terry_rankings function."""
    result = calculate_bradley_terry_rankings(
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


def test_bradley_terry_rating_distribution() -> None:
    """Test that Bradley-Terry ratings have a reasonable distribution.

    This is a regression test for the issue where Bradley-Terry produces
    extreme binary ratings (e.g., 4.00 vs 0.00) when the model fails to
    converge properly.
    """
    # Create a tournament with hierarchy but not extreme dominance
    items = ["A", "B", "C", "D", "E"]
    metric = "test_metric"

    tournament = BradleyTerryTournamentSystem.create(items, metric)

    # Create a tournament with a clear hierarchy
    # Each player beats all players below them in order
    for winner, loser in itertools.pairwise(items):
        tournament = tournament.record_match(
            winner,
            loser,
            MatchResult(winner=MatchWinner.A, explanation=f"{winner} beats {loser}"),
        )

    # Add some variation to avoid complete dominance
    # D beats B once and C beats A once
    tournament = tournament.record_match(
        "D", "B", MatchResult(winner=MatchWinner.A, explanation="Upset: D beats B")
    )
    tournament = tournament.record_match(
        "C", "A", MatchResult(winner=MatchWinner.A, explanation="Upset: C beats A")
    )

    # Check the ratings are reasonably distributed (not extreme values)
    ratings = [r.rating for r in tournament.get_rankings()]

    # The ratings should form a smooth gradient, not binary values
    assert len(set(ratings)) > 2, "Ratings should have more than 2 distinct values"

    # Verify the distribution is reasonable
    assert 0.1 < min(ratings), "Lowest rating should not be too close to zero"

    # Calculate the ratios between consecutive ratings
    # They should be similar but not identical
    ratios = [ratings[i] / ratings[i + 1] for i in range(len(ratings) - 1)]
    assert min(ratios) < max(ratios), (
        "Rating distribution should not be completely uniform"
    )
