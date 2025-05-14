"""Tests for the comparisons module in paper.gpt.evaluate_tournament.comparisons."""

from collections.abc import Sequence

from paper import peerread as pr
from paper.gpt.evaluate_tournament.comparisons import get_comparisons_specs
from paper.gpt.evaluate_tournament.tournament import (
    PaperEvaluationInput,
    get_rationale_eval,
)


class TestGetComparisonSpecs:
    """Tests for the get_comparisons_specs function."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        paper_ids: list[str] = []
        common_papers: dict[str, Sequence[PaperEvaluationInput]] = {}
        item_names = ["model_a", "model_b"]
        metrics = ["clarity"]
        item_indices_pairs = [(0, 1)]

        result = get_comparisons_specs(
            paper_ids, common_papers, item_names, metrics, item_indices_pairs
        )

        assert result == []

    def test_single_paper_single_metric(self, sample_paper: pr.Paper):
        """Test with a single paper and metric."""
        paper_id = sample_paper.id
        paper_ids = [paper_id]
        common_papers = {paper_id: [sample_paper, sample_paper]}
        item_names = ["model_a", "model_b"]
        metrics = ["clarity"]
        item_indices_pairs = [(0, 1)]

        result = get_comparisons_specs(
            paper_ids, common_papers, item_names, metrics, item_indices_pairs
        )

        assert len(result) == 1
        comparison = result[0]
        assert comparison.item_a == "model_a"
        assert comparison.item_b == "model_b"
        assert comparison.paper_id == paper_id
        assert comparison.metric == "clarity"
        assert comparison.paper == sample_paper
        # Rationales should match the sample paper's rationale
        assert comparison.rationale_a == get_rationale_eval(sample_paper)
        assert comparison.rationale_b == get_rationale_eval(sample_paper)

    def test_multiple_papers_multiple_metrics(self, sample_paper: pr.Paper):
        """Test with multiple papers and metrics."""
        paper1 = sample_paper
        paper2 = pr.Paper(
            title="Sample Paper 2",
            authors=["John Smith"],
            abstract="Different abstract",
            approval=None,
            conference="Sample Conference",
            year=2023,
            sections=[],
            references=[],
            reviews=[
                pr.PaperReview(
                    rating=4,
                    confidence=None,
                    rationale="Different rationale",
                )
            ],
        )

        paper_ids = [paper1.id, paper2.id]
        common_papers = {
            paper1.id: [paper1, paper1, paper1],
            paper2.id: [paper2, paper2, paper2],
        }
        item_names = ["model_a", "model_b", "model_c"]
        metrics = ["clarity", "factuality"]
        item_indices_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        result = get_comparisons_specs(
            paper_ids, common_papers, item_names, metrics, item_indices_pairs
        )

        # Should get (2 papers * 2 metrics * 6 item pairs) = 24 comparisons
        assert len(result) == 24

        # Check a few specific values
        # First paper, first metric, first item pair
        first = result[0]
        assert first.paper_id == paper1.id
        assert first.metric == "clarity"
        assert first.item_a == "model_a"
        assert first.item_b == "model_b"

        # Check that all expected combinations exist
        paper_id_counts: dict[str, int] = {}
        metric_counts: dict[str, int] = {}
        item_pair_counts: dict[tuple[str, str], int] = {}

        for spec in result:
            if spec.paper_id in paper_id_counts:
                paper_id_counts[spec.paper_id] += 1
            else:
                paper_id_counts[spec.paper_id] = 1

            if spec.metric in metric_counts:
                metric_counts[spec.metric] += 1
            else:
                metric_counts[spec.metric] = 1

            pair = (spec.item_a, spec.item_b)
            if pair in item_pair_counts:
                item_pair_counts[pair] += 1
            else:
                item_pair_counts[pair] = 1

        assert paper_id_counts == {paper1.id: 12, paper2.id: 12}
        assert metric_counts == {"clarity": 12, "factuality": 12}
        assert item_pair_counts == {
            ("model_a", "model_b"): 4,
            ("model_a", "model_c"): 4,
            ("model_b", "model_a"): 4,
            ("model_b", "model_c"): 4,
            ("model_c", "model_a"): 4,
            ("model_c", "model_b"): 4,
        }

    def test_with_different_rationales(self):
        """Test with papers that have different rationales for each item."""
        # Create sample papers with different rationales
        paper1 = pr.Paper(
            title="Paper with different rationales",
            authors=["Author One"],
            abstract="Testing different rationales",
            approval=None,
            conference="Test Conference",
            year=2023,
            sections=[],
            references=[],
            reviews=[
                pr.PaperReview(
                    rating=5,
                    confidence=None,
                    rationale="Rationale for model A",
                )
            ],
        )

        paper2 = pr.Paper(
            title="Paper with different rationales",
            authors=["Author One"],
            abstract="Testing different rationales",
            approval=None,
            conference="Test Conference",
            year=2023,
            sections=[],
            references=[],
            reviews=[
                pr.PaperReview(
                    rating=5,
                    confidence=None,
                    rationale="Rationale for model B",
                )
            ],
        )

        paper_id = "test_paper_id"
        paper_ids = [paper_id]
        common_papers = {paper_id: [paper1, paper2]}
        item_names = ["model_a", "model_b"]
        metrics = ["clarity"]
        item_indices_pairs = [(0, 1), (1, 0)]

        result = get_comparisons_specs(
            paper_ids, common_papers, item_names, metrics, item_indices_pairs
        )

        assert len(result) == 2

        # Check first comparison (model_a vs model_b)
        first = result[0]
        assert first.item_a == "model_a"
        assert first.item_b == "model_b"
        assert first.rationale_a == get_rationale_eval(paper1)
        assert first.rationale_b == get_rationale_eval(paper2)

        # Check second comparison (model_b vs model_a)
        second = result[1]
        assert second.item_a == "model_b"
        assert second.item_b == "model_a"
        assert second.rationale_a == get_rationale_eval(paper2)
        assert second.rationale_b == get_rationale_eval(paper1)
