"""Common tournament code and data structures.

This module contains the core classes and data structures for running tournaments,
independent of the specific rating algorithm used.
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from enum import StrEnum
from typing import Self

from rich import box
from rich.table import Table

from paper import peerread as pr
from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import PaperWithRelatedSummary
from paper.types import Identifiable, Immutable
from paper.util import render_rich

logger = logging.getLogger(__name__)

TOURNAMENT_METRICS: Mapping[str, str] = {
    "clarity": (
        "How well-written the text is. How easy it is to understand and to follow its"
        " ideas."
    ),
    "faithfulness": (
        "Whether the rationale justifies the novelty label. For example, if the text is"
        " mostly positive, so should the label."
    ),
    "factuality": (
        "Is the rationale grounded correctly in scientific facts from the main and"
        " related papers?"
    ),
    "specificity": (
        "Does the rationale cover information specific to the paper, or does it make"
        " overly generic statements?"
    ),
    "contributions": (
        "Does the rationale effectively compare the main paper with the prior work?"
    ),
}

# Type alias for rationale evaluation inputs
type PaperEvaluationInput = (
    GraphResult | PaperResult | pr.Paper | PaperWithRelatedSummary
)


class MatchWinner(StrEnum):
    """Who won a tournament match."""

    A = "A"
    B = "B"
    TIE = "tie"


class MatchResult(Immutable):
    """Result from pairwise comparison of two rationales by LLM."""

    winner: MatchWinner
    """Who wins the match. The model can only reply A or B, but we use TIE for errors."""
    explanation: str
    """Explanation for the winner evaluation."""


class PlayerMatch(Immutable):
    """Match in player match history."""

    player: str
    opponent: str
    score: float
    explanation: str


class TournamentMatch(Immutable):
    """Entry in tournament match history."""

    player_a: str
    player_b: str
    result: MatchResult


class PlayerRank(Immutable):
    """Player entry in ranking for a given tournament."""

    rank: int
    name: str
    rating: float
    wins: int
    losses: int
    ties: int


class ItemRankStats(Immutable):
    """Statistics for model ranks across metrics."""

    ranks: Sequence[int]
    mean_rank: float
    median_rank: float
    best_rank: int
    worst_rank: int
    metric_ranks: Mapping[str, int]


def calculate_overall_ranks(
    tournaments: Mapping[str, TournamentSystem],
    metrics: Sequence[str],
) -> dict[str, ItemRankStats]:
    """Calculate overall rankings based on average ranks across metrics.

    Args:
        tournaments: Dictionary mapping metric names to tournament systems.
        metrics: List of metrics to consider.

    Returns:
        Dictionary with item names as keys and rank statistics as values.
    """
    # For each item, collect its rank in each tournament
    item_ranks: dict[str, list[int]] = defaultdict(list)
    for metric in metrics:
        for player in tournaments[metric].get_rankings():
            item_ranks[player.name].append(player.rank)

    return {
        item: ItemRankStats(
            ranks=ranks,
            mean_rank=statistics.mean(ranks),
            median_rank=statistics.median(ranks),
            best_rank=min(ranks),
            worst_rank=max(ranks),
            metric_ranks={
                metric: next(
                    player.rank
                    for player in tournaments[metric].get_rankings()
                    if player.name == item
                )
                for metric in metrics
            },
        )
        for item, ranks in item_ranks.items()
    }


def find_common_papers[ID: Identifiable](
    paper_collections: Collection[Collection[ID]],
) -> dict[str, list[ID]]:
    """Find papers that exist in all collections based on ID.

    Args:
        paper_collections: List of lists of paper objects.

    Returns:
        Mapping of paper IDs to list of paper objects from each collection.
    """
    # Extract IDs from each collection
    id_sets = [{p.id for p in papers} for papers in paper_collections]
    # Find IDs common to all collections
    common_ids = set[str].intersection(*id_sets)

    # Group papers by ID
    return {
        paper_id: [
            next(p for p in papers_col if p.id == paper_id)
            for papers_col in paper_collections
        ]
        for paper_id in common_ids
    }


class ComparisonResult(Immutable):
    """Result of comparing two model outputs by LLM."""

    paper: PaperEvaluationInput
    """Full paper data used for the comparison."""
    item_a: str
    """First item name."""
    item_b: str
    """Second item name."""
    rationale_a: str
    """Rationale from item A."""
    rationale_b: str
    """Rationale from item B."""
    metric: str
    """Metric being evaluated."""
    result: MatchResult
    """LLM's comparison result."""


class OverallRankingEntry(Immutable):
    """Overall ranking entry for a model."""

    name: str
    mean_rank: float
    median_rank: float
    best_rank: int
    worst_rank: int
    metric_ranks: Mapping[str, int]


class TournamentSummary(Immutable):
    """Summary of tournament results."""

    item_names: Sequence[str]
    metrics: Sequence[str]
    total_comparisons: int
    metric_rankings: Mapping[str, Sequence[PlayerRank]]
    overall_rankings: Sequence[OverallRankingEntry]


class TournamentResult(Immutable):
    """Full result of all tournaments run."""

    overall_ranks: Mapping[str, ItemRankStats]
    """Item ranks across all tournaments."""
    total_comparisons: int
    """Number of comparisons across items."""
    tournaments: Mapping[str, TournamentSystem]
    """Tournament data for each metric."""


def tournament_summary(
    result: TournamentResult, item_names: Sequence[str], metrics: Sequence[str]
) -> TournamentSummary:
    """Convert internal result to a serializable summary.

    Args:
        result: Tournament result.
        item_names: Names of all items in the tournament.
        metrics: Names of metrics evaluated.

    Returns:
        Serializable tournament summary.
    """
    return TournamentSummary(
        item_names=item_names,
        metrics=metrics,
        total_comparisons=result.total_comparisons,
        metric_rankings={
            metric: [
                PlayerRank(
                    rank=player.rank,
                    name=player.name,
                    rating=player.rating,
                    wins=player.wins,
                    losses=player.losses,
                    ties=player.ties,
                )
                for player in tournament.get_rankings()
            ]
            for metric, tournament in result.tournaments.items()
        },
        overall_rankings=[
            OverallRankingEntry(
                name=name,
                mean_rank=stats.mean_rank,
                median_rank=stats.median_rank,
                best_rank=stats.best_rank,
                worst_rank=stats.worst_rank,
                metric_ranks=stats.metric_ranks,
            )
            for name, stats in sorted(
                result.overall_ranks.items(), key=lambda x: x[1].mean_rank
            )
        ],
    )


def display_tournament_results(
    results: TournamentSummary, markdown: bool = False
) -> str:
    """Format tournament results for display."""
    table = Table(
        title="Tournament Rankings", box=box.MARKDOWN if markdown else box.HEAVY_HEAD
    )

    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Item", style="green")
    table.add_column("Mean Rank", justify="right")
    table.add_column("Median Rank", justify="right")

    metrics = list(results.metric_rankings.keys())
    for metric in metrics:
        table.add_column(metric.capitalize(), justify="right")

    # Add rows for each item's overall ranking
    for i, item in enumerate(results.overall_rankings, 1):
        name = item.name
        mean_rank = f"{item.mean_rank:.2f}"
        median_rank = f"{item.median_rank:.1f}"

        # Get metric-specific ranks
        metric_ranks = [str(item.metric_ranks[m]) for m in metrics]

        table.add_row(str(i), name, mean_rank, median_rank, *metric_ranks)

    return render_rich(table)


def create_tournament_result(
    comparison_results: Collection[ComparisonResult],
    metrics: Sequence[str],
    tournaments: Mapping[str, TournamentSystem],
) -> TournamentResult:
    """Create a tournament result from existing tournament data.

    Args:
        comparison_results: Results of all pairwise comparisons.
        item_names: Names of the items being compared.
        metrics: Metrics that were evaluated.
        tournaments: The existing tournament data for each metric.

    Returns:
        Tournament result with rankings.
    """
    return TournamentResult(
        overall_ranks=calculate_overall_ranks(tournaments, metrics),
        total_comparisons=len(comparison_results),
        tournaments=tournaments,
    )


def count_head_to_head(
    comparison_results: Collection[ComparisonResult],
    item_names: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, dict[tuple[str, str], tuple[int, int, int]]]:
    """Count wins/ties/losses from comparison results.

    Returns:
        Mapping from metric to player head to head.
        Format {metric: {(player_a, player_b): (wins, ties, losses)}}
    """
    h2h_by_metric = {
        metric: {(a, b): (0, 0, 0) for a in item_names for b in item_names if a != b}
        for metric in metrics
    }

    for comp in comparison_results:
        wins_a, ties_a, losses_a = h2h_by_metric[comp.metric][comp.item_a, comp.item_b]
        wins_b, ties_b, losses_b = h2h_by_metric[comp.metric][comp.item_b, comp.item_a]

        match comp.result.winner:
            case MatchWinner.A:
                wins_a += 1
                losses_b += 1
            case MatchWinner.B:
                wins_b += 1
                losses_a += 1
                pass
            case MatchWinner.TIE:
                ties_a += 1
                ties_b += 1

        h2h_by_metric[comp.metric][comp.item_a, comp.item_b] = (
            wins_a,
            ties_a,
            losses_a,
        )
        h2h_by_metric[comp.metric][comp.item_b, comp.item_a] = (
            wins_b,
            ties_b,
            losses_b,
        )

    return h2h_by_metric


def display_head_to_head(
    item_names: Sequence[str],
    metrics: Sequence[str],
    h2h_by_metric: dict[str, dict[tuple[str, str], tuple[int, int, int]]],
) -> str:
    """Display head-to-head comparison results as a table.

    Args:
        item_names: Names of all items being compared.
        metrics: Metrics used for evaluation.
        h2h_by_metric: Mapping of metric to players head-to-head records.

    Returns:
        String representation of the head-to-head table.
    """
    # Create tables for each metric
    tables: list[str] = []

    for metric in metrics:
        table = Table(title=f"Head-to-Head Results: {metric.capitalize()}")

        table.add_column("Player", style="cyan")
        for name in item_names:
            table.add_column(name, justify="center")

        for player_a in item_names:
            row = [player_a]
            for player_b in item_names:
                if player_a == player_b:
                    row.append("—")  # Diagonal cells
                else:
                    wins, ties, losses = h2h_by_metric[metric][player_a, player_b]
                    row.append(f"W:{wins} T:{ties} L:{losses}")

            table.add_row(*row)

        tables.append(render_rich(table))

    return "\n\n".join(tables)


# Abstract base class for tournament systems
class TournamentSystem(Immutable, ABC):
    """Abstract base class for tournament systems.

    This class defines the interface that all tournament systems must implement.
    """

    metric: str

    @classmethod
    @abstractmethod
    def create(cls, item_names: Collection[str], metric: str) -> TournamentSystem:
        """Create a new tournament system.

        This is an abstract method that must be implemented by subclasses.

        Args:
            item_names: Names of the different items being compared.
            metric: The metric this tournament is evaluating.

        Returns:
            A new TournamentSystem instance.
        """
        ...

    @abstractmethod
    def record_match(
        self,
        player_a_name: str,
        player_b_name: str,
        result: MatchResult,
    ) -> TournamentSystem:
        """Record the outcome of a match and update player ratings.

        This is an abstract method that must be implemented by subclasses.

        Args:
            player_a_name: Name of the first player.
            player_b_name: Name of the second player.
            result: The result of the comparison.

        Returns:
            A new TournamentSystem with updated state.
        """
        ...

    @abstractmethod
    def get_rankings(self) -> list[PlayerRank]:
        """Get rankings for all players.

        This is an abstract method that must be implemented by subclasses.

        Returns:
            A list of player rankings.
        """
        ...


class InputFileType(StrEnum):
    """Types of input data formats."""

    RAW = "raw"
    """Original dataset paper: `peerread.Paper`"""
    GRAPH = "graph"
    """Output of `gpt.evaluate_paper_graph`: `PromptResult[GraphResult]`"""
    PAPER = "paper"
    """Output of `gpt.evaluate_paper_scimon`: `PromptResult[PaperResult]`"""
    SUMM = "summ"
    """Output of `gpt.summarise_related_peter`: `PromptResult[PaperWithRelatedSummary]`"""

    @classmethod
    def from_dirty(cls, type_: str) -> Self:
        """Create instance by cleaning up `type_` by stripping and lowercasing.

        Use when `type_` comes from a potentially dirty source, such as a CLI argument.

        Raises:
            ValueError if the type is invalid.
        """
        return cls(type_.strip().lower())


def get_rationale_eval(paper: PaperEvaluationInput) -> str:
    """Get paper rationale to be evaluated.

    For result types (GraphResult/PaperResult), gets the model predicted rationale. For
    original types (pr.Paper/PaperWithRelatedSummary), gets the gold rationale.
    """

    match paper:
        case GraphResult() | PaperResult():
            return paper.rationale_pred
        case pr.Paper() | PaperWithRelatedSummary():
            return paper.rationale
