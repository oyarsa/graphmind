"""Tests for summary cleaning post-processing."""

import pytest

from paper.single_paper.summary_cleaning import clean_summary


class TestCleanSummary:
    """Tests for the clean_summary function."""

    # -------------------------------------------------------------------------
    # Formulaic prefix patterns (most common)
    # -------------------------------------------------------------------------

    def test_supporting_paper_demonstrates(self) -> None:
        text = (
            "The supporting paper demonstrates that event co-occurrences can "
            "enhance event argument extraction."
        )
        result = clean_summary(text)
        assert result == ("Event co-occurrences can enhance event argument extraction.")

    def test_contrasting_paper_emphasizes(self) -> None:
        text = (
            "The contrasting paper emphasizes a binary classification approach "
            "to causal knowledge extraction, focusing on sequence tagging."
        )
        result = clean_summary(text)
        assert result == (
            "A binary classification approach to causal knowledge extraction, "
            "focusing on sequence tagging."
        )

    def test_contrasting_paper_introduces(self) -> None:
        text = (
            "The contrasting paper introduces a novel approach to Event "
            "Causality Identification (ECI) by focusing on context words."
        )
        result = clean_summary(text)
        assert result == (
            "A novel approach to Event Causality Identification (ECI) by "
            "focusing on context words."
        )

    def test_second_paper_contrasts_with(self) -> None:
        text = (
            "The second paper contrasts with the first by focusing on a "
            "classification framework for Event Causality Identification."
        )
        result = clean_summary(text)
        assert result == (
            "Focusing on a classification framework for Event Causality Identification."
        )

    def test_supporting_paper_corroborates(self) -> None:
        text = (
            "The supporting paper corroborates the findings by addressing "
            "similar challenges in event extraction."
        )
        result = clean_summary(text)
        assert result == (
            "The findings by addressing similar challenges in event extraction."
        )

    def test_related_paper_shows(self) -> None:
        text = "The related paper shows that learned metrics outperform BLEU."
        result = clean_summary(text)
        assert result == "Learned metrics outperform BLEU."

    def test_first_paper_highlights(self) -> None:
        text = (
            "The first paper highlights the importance of semantic boundaries "
            "in extraction tasks."
        )
        result = clean_summary(text)
        assert result == ("The importance of semantic boundaries in extraction tasks.")

    # -------------------------------------------------------------------------
    # Findings prefix patterns
    # -------------------------------------------------------------------------

    def test_findings_from_supporting_paper_highlight(self) -> None:
        text = (
            "The findings from the supporting paper highlight the importance "
            "of addressing semantic boundaries and interference in extraction."
        )
        result = clean_summary(text)
        assert result == (
            "The importance of addressing semantic boundaries and interference "
            "in extraction."
        )

    def test_findings_from_contrasting_paper_show_that(self) -> None:
        text = (
            "The findings from the contrasting paper show that distant "
            "supervision can effectively augment training data."
        )
        result = clean_summary(text)
        assert result == ("Distant supervision can effectively augment training data.")

    def test_findings_of_the_related_paper_demonstrate(self) -> None:
        text = (
            "The findings of the related paper demonstrate that co-occurrences "
            "improve argument extraction accuracy."
        )
        result = clean_summary(text)
        assert result == ("Co-occurrences improve argument extraction accuracy.")

    # -------------------------------------------------------------------------
    # Development prefix patterns
    # -------------------------------------------------------------------------

    def test_development_of_causalbank_provides(self) -> None:
        text = (
            "The development of CausalBank in the supporting paper provides "
            "a large-scale resource of causal patterns that can enhance training."
        )
        result = clean_summary(text)
        assert result == (
            "A large-scale resource of causal patterns that can enhance training."
        )

    def test_introduction_of_method_in_contrasting_paper_demonstrates(self) -> None:
        text = (
            "The introduction of the ORCA method in the contrasting paper "
            "demonstrates superior performance on benchmark datasets."
        )
        result = clean_summary(text)
        assert result == "Superior performance on benchmark datasets."

    # -------------------------------------------------------------------------
    # Text that should NOT be modified
    # -------------------------------------------------------------------------

    def test_no_change_for_natural_opening(self) -> None:
        text = (
            "BLEURT's development of a learned evaluation metric based on BERT "
            "aligns with the weak supervision method proposed."
        )
        result = clean_summary(text)
        assert result == text

    def test_no_change_for_direct_statement(self) -> None:
        text = (
            "Event co-occurrences can enhance event argument extraction, which "
            "aligns with the emphasis on improving causal event extraction."
        )
        result = clean_summary(text)
        assert result == text

    def test_no_change_for_comparison_opening(self) -> None:
        text = (
            "Weak Reward Model focuses on improving causal event extraction "
            "through reinforcement learning, while KnowDis emphasizes data "
            "augmentation using distant supervision."
        )
        result = clean_summary(text)
        assert result == text

    def test_no_change_for_specific_method_opening(self) -> None:
        text = (
            "The retrieval-augmented code generation framework, Code4UIE, "
            "supports the findings by demonstrating the effectiveness of "
            "using structured knowledge extraction."
        )
        result = clean_summary(text)
        assert result == text

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_empty_string(self) -> None:
        assert clean_summary("") == ""

    def test_case_insensitivity(self) -> None:
        text = "THE SUPPORTING PAPER DEMONSTRATES that methods work."
        result = clean_summary(text)
        assert result == "Methods work."

    def test_with_appositive(self) -> None:
        text = (
            "The supporting paper, 'CausalBank', provides a large-scale "
            "resource for causal pattern extraction."
        )
        result = clean_summary(text)
        assert result == ("A large-scale resource for causal pattern extraction.")


@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        # American vs British spelling
        (
            "The contrasting paper emphasizes data augmentation.",
            "Data augmentation.",
        ),
        (
            "The contrasting paper emphasises data augmentation.",
            "Data augmentation.",
        ),
        # Different paper types
        (
            "The prior paper demonstrates effective results.",
            "Effective results.",
        ),
        (
            "The target paper shows improved metrics.",
            "Improved metrics.",
        ),
        (
            "The main paper provides a novel approach.",
            "A novel approach.",
        ),
    ],
)
def test_parametrised_cases(input_text: str, expected: str) -> None:
    """Test various input patterns produce expected outputs."""
    assert clean_summary(input_text) == expected
