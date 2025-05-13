"""Elo rating system implementation.

This module implements the Elo rating system for tournament comparisons,
including a Multi-Elo (MElo) approach that runs multiple tournaments.
"""

from __future__ import annotations

import logging
import random
import statistics
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

# Elo rating constants
DEFAULT_ELO = 1200  # Starting rating for all players
K_FACTOR = 32  # How much ratings can change in a single match
EXPECTED_SCORE_DIVISOR = 400  # For expected score calculation
MELO_DEFAULT_TRIALS = 10  # How many different Elo trials to run


class EloPlayer(Immutable):
    """Represents a player in the Elo rating system."""

    name: str
    rating: float
    wins: int
    losses: int
    ties: int
    match_history: Sequence[PlayerMatch]

    @classmethod
    def fresh(cls, name: str) -> EloPlayer:
        """Create fresh player without rating or matches."""
        return EloPlayer(
            name=name, rating=DEFAULT_ELO, wins=0, losses=0, ties=0, match_history=()
        )

    def play_match(
        self,
        opponent: str,
        expected_score: float,
        actual_score: float,
        explanation: str,
    ) -> EloPlayer:
        """Play match against `opponent`, updating the match record and rating."""
        new_match = PlayerMatch(
            player=self.name,
            opponent=opponent,
            score=actual_score,
            explanation=explanation,
        )

        return EloPlayer(
            wins=self.wins + (1 if actual_score == 1.0 else 0),
            ties=self.ties + (1 if actual_score == 0.5 else 0),
            losses=self.losses + (1 if actual_score == 0.0 else 0),
            match_history=(*self.match_history, new_match),
            rating=_update_elo_rating(self.rating, actual_score, expected_score),
            name=self.name,
        )


def _update_elo_rating(
    current_rating: float, actual_score: float, expected_score: float
) -> float:
    """Update Elo rating based on match result.

    Args:
        current_rating: Current rating of the player.
        actual_score: Actual score of the player (0.0-1.0).
        expected_score: Expected score of the player (0.0-1.0).

    Returns:
        Updated Elo rating.
    """
    return current_rating + K_FACTOR * (actual_score - expected_score)


def _elo_expected_probabilities(
    player_a: EloPlayer, player_b: EloPlayer
) -> tuple[float, float]:
    """Calculate expected score for player A against player B.

    Args:
        player_a: First player.
        player_b: Second player.

    Returns:
        Tuple with expected probabilities of (player A, player B) winning in [0, 1].
    """
    expected_a = 1.0 / (
        1.0 + 10.0 ** ((player_b.rating - player_a.rating) / EXPECTED_SCORE_DIVISOR)
    )
    expected_b = 1 - expected_a
    return expected_a, expected_b


class EloTournamentSystem(TournamentSystem):
    """Manages a tournament for rationale comparisons using Elo ratings."""

    metric: str
    players: Mapping[str, EloPlayer]
    matches: Sequence[TournamentMatch]

    @classmethod
    def create(cls, item_names: Collection[str], metric: str) -> EloTournamentSystem:
        """Create a new tournament system from the players (items) names and metrics.

        Args:
            item_names: Names of the different items being compared.
            metric: The metric this tournament is evaluating.

        Returns:
            A new EloTournamentSystem instance.
        """
        return cls(
            metric=metric,
            players={name: EloPlayer.fresh(name=name) for name in item_names},
            matches=(),
        )

    def record_match(
        self,
        player_a_name: str,
        player_b_name: str,
        result: MatchResult,
    ) -> EloTournamentSystem:
        """Record the outcome of a match and update player ratings.

        Args:
            player_a_name: Name of the first player.
            player_b_name: Name of the second player.
            result: The result of the comparison.

        Returns:
            A new EloTournamentSystem with updated state.
        """
        player_a = self.players[player_a_name]
        player_b = self.players[player_b_name]

        expected_a, expected_b = _elo_expected_probabilities(player_a, player_b)

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
                player_b_name, expected_a, actual_a, result.explanation
            ),
            player_b_name: player_b.play_match(
                player_a_name, expected_b, actual_b, result.explanation
            ),
        }

        new_match = TournamentMatch(
            player_a=player_a_name, player_b=player_b_name, result=result
        )
        return EloTournamentSystem(
            metric=self.metric,
            players=updated_players,
            matches=(*self.matches, new_match),
        )

    def get_rankings(self) -> list[PlayerRank]:
        """Get rankings for all players."""
        players_sorted_by_rating = sorted(
            self.players.values(), key=lambda p: p.rating, reverse=True
        )

        return [
            PlayerRank(
                rank=i,
                name=player.name,
                rating=player.rating,
                wins=player.wins,
                losses=player.losses,
                ties=player.ties,
            )
            for i, player in enumerate(players_sorted_by_rating, 1)
        ]


class TournamentManager(Immutable):
    """Manage multiple tournaments, one per metric."""

    tournaments: Mapping[str, TournamentSystem]

    @classmethod
    def create(
        cls, metrics: Sequence[str], item_names: Sequence[str]
    ) -> TournamentManager:
        """Create manager with empty tournaments for each metric."""
        return cls(
            tournaments={
                metric: EloTournamentSystem.create(item_names, metric)
                for metric in metrics
            }
        )

    def record_match(
        self, metric: str, player_a: str, player_b: str, result: MatchResult
    ) -> TournamentManager:
        """Record match result between players A and B in `metric`."""
        return TournamentManager(
            tournaments={
                **self.tournaments,
                metric: self.tournaments[metric].record_match(
                    player_a, player_b, result
                ),
            }
        )


def calculate_elo_rankings(
    comparison_results: Collection[ComparisonResult],
    item_names: Sequence[str],
    metrics: Sequence[str],
) -> TournamentResult:
    """Calculate Elo rankings from comparison results.

    Args:
        comparison_results: Results of all pairwise comparisons.
        item_names: Names of the items being compared.
        metrics: Metrics that were evaluated.

    Returns:
        Tournament results with Elo rankings.
    """
    tour_manager = TournamentManager.create(metrics, item_names)

    logger.info("Calculating Elo rankings from %d comparisons", len(comparison_results))
    for comparison in comparison_results:
        tour_manager = tour_manager.record_match(
            comparison.metric,
            comparison.item_a,
            comparison.item_b,
            comparison.result,
        )

    return create_tournament_result(
        comparison_results=comparison_results,
        metrics=metrics,
        tournaments=tour_manager.tournaments,
    )


def calculate_melo_rankings(
    comparison_results: Collection[ComparisonResult],
    item_names: Sequence[str],
    metrics: Sequence[str],
    num_trials: int,
    seed: int,
) -> TournamentResult:
    """Calculate Multi-Elo rankings from comparison results.

    Runs multiple Elo tournaments with different random orderings and averages
    the final ratings.

    Args:
        comparison_results: Results of all pairwise comparisons.
        item_names: Names of the items being compared.
        metrics: Metrics that were evaluated.
        num_trials: Number of tournaments to run with different orderings.
        seed: Random seed for reproducibility.

    Returns:
        Tournament results with Multi-Elo rankings.
    """
    # Group comparisons by metric to simplify the multi-tournament process
    comparisons_by_metric: dict[str, list[ComparisonResult]] = defaultdict(list)
    for comp in comparison_results:
        comparisons_by_metric[comp.metric].append(comp)

    # Initialize structures to average results across trials
    all_ratings: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_ranks: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_wins: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_losses: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_ties: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    random_gen = random.Random(seed)

    logger.info(f"Running {num_trials} tournaments with different orderings")
    for _ in range(num_trials):
        trial_seed = random_gen.randint(0, 10000)
        trial_random = random.Random(trial_seed)

        tour_manager = TournamentManager.create(metrics, item_names)

        # For each metric, shuffle the comparisons differently
        for metric in metrics:
            metric_comparisons = comparisons_by_metric[metric].copy()
            trial_random.shuffle(metric_comparisons)

            # Process the shuffled comparisons
            for cmp in metric_comparisons:
                tour_manager = tour_manager.record_match(
                    cmp.metric, cmp.item_a, cmp.item_b, cmp.result
                )

            # Record each player's result for this metric in this trial
            for player in tour_manager.tournaments[metric].get_rankings():
                all_ratings[metric][player.name].append(player.rating)
                all_ranks[metric][player.name].append(player.rank)
                all_wins[metric][player.name].append(player.wins)
                all_losses[metric][player.name].append(player.losses)
                all_ties[metric][player.name].append(player.ties)

    # Create final tournaments with averaged results
    final_tournaments: dict[str, EloTournamentSystem] = {}
    for metric in metrics:
        # For each item, create a new player with averaged statistics
        updated_players: dict[str, EloPlayer] = {}
        for name in item_names:
            # Create a new player with averaged values
            updated_players[name] = EloPlayer(
                name=name,
                rating=statistics.mean(all_ratings[metric][name]),
                wins=int(statistics.mean(all_wins[metric][name])),
                losses=int(statistics.mean(all_losses[metric][name])),
                ties=int(statistics.mean(all_ties[metric][name])),
                match_history=(),  # Not a real player.
            )

        # Create a tournament with averaged values
        final_tournaments[metric] = EloTournamentSystem(
            metric=metric,
            players=updated_players,
            matches=(),  # Not a real tournament.
        )

    return create_tournament_result(
        comparison_results=comparison_results,
        metrics=metrics,
        tournaments=final_tournaments,
    )
