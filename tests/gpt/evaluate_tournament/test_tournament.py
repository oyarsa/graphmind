"""Tests for tournament functionality in paper.gpt.evaluate_tournament.tournament."""

from __future__ import annotations
from itertools import product
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
    find_common_papers,
    count_head_to_head,
)
from paper import peerread as pr


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


class MockPaperWithId:
    """Mock paper class with an id property."""

    def __init__(self, paper_id: str, rationale: str) -> None:
        self.id = paper_id
        self.rationale = rationale


class TestFindCommonPapers:
    """Tests for the find_common_papers function."""

    def test_find_common_papers_with_empty_collections(self):
        """Test with empty collections."""
        empty: list[MockPaperWithId] = []
        result = find_common_papers([empty, empty])
        assert result == {}

    def test_find_common_papers_with_no_overlap(self):
        """Test with collections that have no papers in common."""
        collection1 = [
            MockPaperWithId("paper1", "paper 1 rationale 1"),
            MockPaperWithId("paper2", "paper 2 rationale 1"),
        ]
        collection2 = [
            MockPaperWithId("paper3", "paper 3 rationale 2"),
            MockPaperWithId("paper4", "paper 4 rationale 2"),
        ]

        result = find_common_papers([collection1, collection2])
        assert result == {}

    def test_find_common_papers_with_partial_overlap(self):
        """Test with collections that have some papers in common."""
        collection1 = [
            MockPaperWithId("paper1", "paper1 rationale 1"),
            MockPaperWithId("paper2", "paper2 rationale 1"),
        ]
        collection2 = [
            MockPaperWithId("paper2", "paper2 rationale 2"),
            MockPaperWithId("paper3", "paper3 rationale 2"),
        ]

        result = find_common_papers([collection1, collection2])
        assert len(result) == 1
        assert "paper2" in result
        assert len(result["paper2"]) == 2

        p1, p2 = result["paper2"]
        assert p1.id == "paper2"
        assert p2.id == "paper2"
        assert p1.rationale != p2.rationale

    def test_find_common_papers_with_multiple_collections(self):
        """Test with multiple collections."""
        collection1 = [
            MockPaperWithId("paper1", "paper1 rationale 1"),
            MockPaperWithId("paper2", "paper2 rationale 1"),
            MockPaperWithId("paper3", "paper3 rationale 1"),
        ]
        collection2 = [
            MockPaperWithId("paper2", "paper2 rationale 2"),
            MockPaperWithId("paper3", "paper3 rationale 2"),
            MockPaperWithId("paper4", "paper4 rationale 2"),
        ]
        collection3 = [
            MockPaperWithId("paper3", "paper3 rationale 3"),
            MockPaperWithId("paper4", "paper4 rationale 3"),
            MockPaperWithId("paper5", "paper5 rationale 3"),
        ]

        result = find_common_papers([collection1, collection2, collection3])
        assert len(result) == 1
        assert "paper3" in result
        assert len(result["paper3"]) == 3
        assert all(paper.id == "paper3" for paper in result["paper3"])
        assert len(set(p.rationale for p in result["paper3"])) > 1

    def test_find_common_papers_with_all_common(self):
        """Test with collections where all papers are common."""
        collection1 = [
            MockPaperWithId("paper1", "paper1 rationale 1"),
            MockPaperWithId("paper2", "paper2 rationale 1"),
        ]
        collection2 = [
            MockPaperWithId("paper1", "paper1 rationale 2"),
            MockPaperWithId("paper2", "paper2 rationale 2"),
        ]

        result = find_common_papers([collection1, collection2])
        assert len(result) == 2
        assert "paper1" in result and "paper2" in result
        assert all(len(papers) == 2 for papers in result.values())

    def test_with_actual_paper_type(self, sample_paper: pr.Paper):
        """Test with a real PeerRead paper object."""
        # Create 2 collections, each with our test paper
        paper_id = sample_paper.id
        collections = [
            [sample_paper],  # pr.Paper
            [sample_paper],  # pr.Paper
        ]

        result = find_common_papers(collections)

        # Verify results
        assert len(result) == 1
        assert paper_id in result
        assert len(result[paper_id]) == 2


def test_count_head_to_head(
    sample_player_names: list[str],
    sample_metrics: list[str],
    sample_comparison_results: list[ComparisonResult],
):
    """Test count_head_to_head function."""
    h2h = count_head_to_head(
        sample_comparison_results, sample_player_names, sample_metrics
    )

    # Check structure
    assert set(h2h.keys()) == set(sample_metrics)

    for metric in sample_metrics:
        # Each metric should have h2h data for each player pair
        expected_pairs = {
            (a, b) for a in sample_player_names for b in sample_player_names if a != b
        }
        assert set(h2h[metric].keys()) == expected_pairs

        for pair in h2h[metric]:
            player_a, player_b = pair
            wins_a, ties_a, losses_a = h2h[metric][player_a, player_b]
            wins_b, ties_b, losses_b = h2h[metric][player_b, player_a]

            # Wins for A should equal losses for B
            assert wins_a == losses_b

            # Ties should be the same in both directions
            assert ties_a == ties_b

            # Losses for A should equal wins for B
            assert losses_a == wins_b

            # The total of wins, ties, and losses should be at most
            # the number of comparison results for this pair
            assert wins_a + ties_a + losses_a <= len(sample_comparison_results)


def test_count_head_to_head_empty_results():
    """Test count_head_to_head with empty comparison results."""
    player_names = ["model_a", "model_b"]
    metrics = ["clarity"]
    empty_results: list[ComparisonResult] = []

    h2h = count_head_to_head(empty_results, player_names, metrics)

    # Check structure is created even with empty results
    assert set(h2h.keys()) == set(metrics)

    # All head-to-head records should be (0, 0, 0)
    for metric, player_a, player_b in product(metrics, player_names, player_names):
        if player_a != player_b:
            assert h2h[metric][player_a, player_b] == (0, 0, 0)


def test_count_head_to_head_with_ties(
    sample_paper: pr.Paper, sample_player_names: list[str]
):
    """Test count_head_to_head handling of ties."""
    # Create comparison results with ties
    metric = "clarity"
    player_a = sample_player_names[0]
    player_b = sample_player_names[1]

    # Create a ComparisonResult with a tie
    tie_result = ComparisonResult(
        paper=sample_paper,
        item_a=player_a,
        item_b=player_b,
        rationale_a="Rationale for A",
        rationale_b="Rationale for B",
        metric=metric,
        result=MatchResult(winner=MatchWinner.TIE, explanation="It's a tie"),
    )

    h2h = count_head_to_head([tie_result], sample_player_names, [metric])

    # Check that ties are counted correctly
    assert h2h[metric][(player_a, player_b)] == (0, 1, 0)  # wins, ties, losses
    assert h2h[metric][(player_b, player_a)] == (0, 1, 0)  # wins, ties, losses
