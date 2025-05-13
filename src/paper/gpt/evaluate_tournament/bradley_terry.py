"""Bradley-Terry rating system implementation.

This module implements the Bradley-Terry model for tournament comparisons,
which estimates the strength of each player based on pairwise comparisons.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence

from paper.gpt.evaluate_tournament.tournament import (
    ComparisonResult,
    MatchResult,
    MatchWinner,
    PlayerMatch,
    PlayerRank,
    TournamentMatch,
    TournamentResult,
    TournamentSystem,
    create_tournament_result,
)
from paper.types import Immutable

logger = logging.getLogger(__name__)

# Bradley-Terry model constants
DEFAULT_BT_STRENGTH = 1.0  # Initial strength parameter for all players
BT_CONVERGENCE_THRESHOLD = 1e-6  # Threshold for convergence in MLE
BT_MAX_ITERATIONS = 100  # Maximum iterations for MLE algorithm


class BradleyTerryPlayer(Immutable):
    """Represents a player in the Bradley-Terry rating system."""

    name: str
    strength: float  # Strength parameter θᵢ
    wins: int
    losses: int
    ties: int
    match_history: Sequence[PlayerMatch]

    @classmethod
    def fresh(cls, name: str) -> BradleyTerryPlayer:
        """Create fresh player with initial strength and no matches."""
        return BradleyTerryPlayer(
            name=name,
            strength=DEFAULT_BT_STRENGTH,
            wins=0,
            losses=0,
            ties=0,
            match_history=(),
        )

    def play_match(
        self,
        opponent: str,
        actual_score: float,
        explanation: str,
    ) -> BradleyTerryPlayer:
        """Record a match against an opponent.

        Unlike Elo, Bradley-Terry doesn't update strengths immediately; they are
        recomputed globally after all matches.

        Args:
            opponent: The name of the opponent.
            actual_score: The actual score (1.0=win, 0.5=tie, 0.0=loss).
            explanation: The explanation for the match result.

        Returns:
            Updated player with new match history.
        """
        new_match = PlayerMatch(
            player=self.name,
            opponent=opponent,
            score=actual_score,
            explanation=explanation,
        )

        return BradleyTerryPlayer(
            name=self.name,
            strength=self.strength,  # Unchanged until global recomputation
            wins=self.wins + (1 if actual_score == 1.0 else 0),
            ties=self.ties + (1 if actual_score == 0.5 else 0),
            losses=self.losses + (1 if actual_score == 0.0 else 0),
            match_history=(*self.match_history, new_match),
        )

    def set_strength(self, new_strength: float) -> BradleyTerryPlayer:
        """Create a new player with updated strength parameter but same history.

        Args:
            new_strength: The new strength parameter.

        Returns:
            Updated player with new strength value.
        """
        return BradleyTerryPlayer(
            name=self.name,
            strength=new_strength,
            wins=self.wins,
            losses=self.losses,
            ties=self.ties,
            match_history=self.match_history,
        )


def _bt_update_strengths(
    players: Mapping[str, BradleyTerryPlayer],
    win_counts: Mapping[tuple[str, str], float],
    match_counts: Mapping[tuple[str, str], float],
) -> dict[str, BradleyTerryPlayer]:
    """Update player strengths using the Bradley-Terry model's maximum likelihood estimation.

    This is an iterative algorithm that converges to the maximum likelihood estimates
    of the player strengths given the observed match outcomes.

    Args:
        players: Current mapping of player names to player objects.
        win_counts: Mapping of (player, opponent) pairs to count of wins (1.0 for win, 0.5 for tie).
        match_counts: Mapping of (player, opponent) pairs to count of matches played.

    Returns:
        Updated mapping of player names to player objects with new strength values.
    """
    player_names = list(players.keys())
    n_players = len(player_names)

    current_strengths = {name: players[name].strength for name in player_names}

    # Iterative MLE algorithm to estimate probabilities
    for _ in range(BT_MAX_ITERATIONS):
        new_strengths: dict[str, float] = {}
        max_diff = 0.0

        for name in player_names:
            # Number of matches player has played
            total_matches = sum(
                match_counts.get((name, opponent), 0)
                for opponent in player_names
                if opponent != name
            )

            if total_matches == 0:
                # If player hasn't played any matches, keep strength unchanged
                new_strengths[name] = current_strengths[name]
                continue

            # Total wins for player i
            total_wins = sum(
                win_counts.get((name, opponent), 0)
                for opponent in player_names
                if opponent != name
            )

            # Expected wins based on current parameters
            expected_wins = 0.0
            for opponent in player_names:
                if opponent == name:
                    continue

                matches = match_counts.get((name, opponent), 0)
                if matches > 0:
                    # Probability of winning against this opponent
                    opponent_strength = current_strengths[opponent]
                    # Prevent division by zero by ensuring strengths are positive
                    player_strength = max(current_strengths[name], 1e-10)
                    opp_strength = max(opponent_strength, 1e-10)
                    p_win = player_strength / (player_strength + opp_strength)
                    expected_wins += matches * p_win

            # Avoid division by zero
            if expected_wins < 1e-10:
                new_strengths[name] = current_strengths[name]
                continue

            # Update strength
            new_strength = current_strengths[name] * total_wins / expected_wins
            new_strengths[name] = new_strength

            # Track largest change
            diff = abs(new_strength - current_strengths[name])
            max_diff = max(max_diff, diff)

        # Normalize to avoid numerical issues
        strength_values = list(new_strengths.values())
        sum_strengths = sum(strength_values)
        if sum_strengths > 0:
            scale_factor = n_players / sum_strengths
            for name_str in list(new_strengths.keys()):
                new_strengths[name_str] = new_strengths[name_str] * scale_factor

        # Check for convergence
        if max_diff < BT_CONVERGENCE_THRESHOLD:
            break

        current_strengths = new_strengths

    # Create new player objects with updated strengths
    return {
        name: players[name].set_strength(current_strengths[name])
        for name in player_names
    }


class BradleyTerryTournamentSystem(TournamentSystem):
    """Manages a tournament for rationale comparisons using the Bradley-Terry model."""

    metric: str
    players: Mapping[str, BradleyTerryPlayer]
    matches: Sequence[TournamentMatch]

    @classmethod
    def create(
        cls, item_names: Collection[str], metric: str
    ) -> BradleyTerryTournamentSystem:
        """Create a new tournament system from the players (items) names and metrics.

        Args:
            item_names: Names of the different items being compared.
            metric: The metric this tournament is evaluating.

        Returns:
            A new BradleyTerryTournamentSystem instance.
        """
        return cls(
            metric=metric,
            players={name: BradleyTerryPlayer.fresh(name=name) for name in item_names},
            matches=(),
        )

    def record_match(
        self,
        player_a_name: str,
        player_b_name: str,
        result: MatchResult,
    ) -> BradleyTerryTournamentSystem:
        """Record the outcome of a match and add it to match history.

        Unlike Elo, Bradley-Terry doesn't update ratings immediately after each match.
        Instead, all matches are recorded and then strength parameters are computed
        globally after all match outcomes are known.

        Args:
            player_a_name: Name of the first player.
            player_b_name: Name of the second player.
            result: The result of the comparison.

        Returns:
            A new BradleyTerryTournamentSystem with the match recorded.
        """
        player_a = self.players[player_a_name]
        player_b = self.players[player_b_name]

        # Determine actual scores from the result
        match result.winner:
            case MatchWinner.A:
                actual_a, actual_b = 1.0, 0.0
            case MatchWinner.B:
                actual_a, actual_b = 0.0, 1.0
            case MatchWinner.TIE:
                actual_a, actual_b = 0.5, 0.5

        updated_players = {
            **self.players,
            player_a_name: player_a.play_match(
                player_b_name, actual_a, result.explanation
            ),
            player_b_name: player_b.play_match(
                player_a_name, actual_b, result.explanation
            ),
        }

        new_match = TournamentMatch(
            player_a=player_a_name, player_b=player_b_name, result=result
        )

        system = BradleyTerryTournamentSystem(
            metric=self.metric,
            players=updated_players,
            matches=(*self.matches, new_match),
        )

        # After recording all matches, update all player strengths at once
        return system._update_all_strengths()

    def _update_all_strengths(self) -> BradleyTerryTournamentSystem:
        """Update the strength parameters of all players using the Bradley-Terry model.

        This is called after recording matches to globally update all player strengths.

        Returns:
            A new BradleyTerryTournamentSystem with updated player strengths.
        """
        # Compute win counts and match counts for all player pairs
        win_counts: dict[tuple[str, str], float] = {}
        match_counts: dict[tuple[str, str], float] = {}

        for match in self.matches:
            player_a = match.player_a
            player_b = match.player_b

            match_counts[player_a, player_b] = (
                match_counts.get((player_a, player_b), 0) + 1
            )
            match_counts[player_b, player_a] = (
                match_counts.get((player_b, player_a), 0) + 1
            )

            # Record win counts based on match result
            a_vs_b = win_counts.get((player_a, player_b), 0)
            b_vs_a = win_counts.get((player_b, player_a), 0)

            match match.result.winner:
                case MatchWinner.A:
                    win_counts[player_a, player_b] = a_vs_b + 1
                case MatchWinner.B:
                    win_counts[player_b, player_a] = b_vs_a + 1
                case MatchWinner.TIE:
                    win_counts[player_a, player_b] = a_vs_b + 0.5
                    win_counts[player_b, player_a] = b_vs_a + 0.5

        # Update player strengths using maximum likelihood estimation
        updated_players = _bt_update_strengths(self.players, win_counts, match_counts)

        return BradleyTerryTournamentSystem(
            metric=self.metric,
            players=updated_players,
            matches=self.matches,
        )

    def get_rankings(self) -> list[PlayerRank]:
        """Get rankings for all players based on their strength parameters."""
        players_sorted_by_strength = sorted(
            self.players.values(), key=lambda p: p.strength, reverse=True
        )

        return [
            PlayerRank(
                rank=i,
                name=player.name,
                rating=player.strength,  # Use strength as rating for output
                wins=player.wins,
                losses=player.losses,
                ties=player.ties,
            )
            for i, player in enumerate(players_sorted_by_strength, 1)
        ]


def calculate_bradley_terry_rankings(
    comparison_results: Collection[ComparisonResult],
    item_names: Sequence[str],
    metrics: Sequence[str],
) -> TournamentResult:
    """Calculate rankings using the Bradley-Terry model from comparison results.

    The Bradley-Terry model computes strength parameters for each player
    based on the outcomes of all pairwise comparisons. It estimates the
    probability of player i beating player j as θᵢ/(θᵢ + θⱼ) where θ are
    the strength parameters.

    Args:
        comparison_results: Results of all pairwise comparisons.
        item_names: Names of the items being compared.
        metrics: Metrics that were evaluated.

    Returns:
        Tournament results with Bradley-Terry rankings.
    """
    logger.info(
        "Calculating Bradley-Terry rankings from %d comparisons",
        len(comparison_results),
    )

    # Group comparisons by metric
    comparisons_by_metric: dict[str, list[ComparisonResult]] = defaultdict(list)
    for comp in comparison_results:
        comparisons_by_metric[comp.metric].append(comp)

    # Create a tournament system for each metric
    bt_tournaments: dict[str, BradleyTerryTournamentSystem] = {}
    for metric in metrics:
        tournament = BradleyTerryTournamentSystem.create(item_names, metric)

        for comparison in comparisons_by_metric[metric]:
            tournament = tournament.record_match(
                comparison.item_a,
                comparison.item_b,
                comparison.result,
            )

        bt_tournaments[metric] = tournament

    return create_tournament_result(
        comparison_results=comparison_results,
        metrics=metrics,
        tournaments=bt_tournaments,
    )
