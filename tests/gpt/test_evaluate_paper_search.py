"""Tests for the evaluate_paper_search module."""

from paper.gpt.evaluate_paper_search import parse_result


class TestParseResult:
    """Test the parse_result function with various output formats."""

    def test_parse_result_with_bold_label(self):
        """Test parsing with **Label: 1** format."""
        text = """The paper is novel.

        **Label: 1**

        **References:**
        1. Some paper"""

        result = parse_result(text)
        assert result.label == 1
        assert "The paper is novel." in result.rationale
        assert "**References:**" in result.rationale
        assert "1. Some paper" in result.rationale

    def test_parse_result_with_bold_label_colon_outside(self):
        """Test parsing with **Label**: 0 format."""
        text = """The paper is not novel.

        **Label**: 0

        References:
        1. Some paper"""

        result = parse_result(text)
        assert result.label == 0
        assert "The paper is not novel." in result.rationale

    def test_parse_result_with_plain_label(self):
        """Test parsing with plain Label: 1 format."""
        text = """The paper is novel.

        Label: 1

        References:
        1. Some paper"""

        result = parse_result(text)
        assert result.label == 1
        assert "The paper is novel." in result.rationale

    def test_parse_result_with_rationale_prefix(self):
        """Test parsing with Rationale: prefix."""
        text = """Rationale: The paper introduces novel concepts.

        Label: 1"""

        result = parse_result(text)
        assert result.label == 1
        assert result.rationale == "Rationale: The paper introduces novel concepts."

    def test_parse_result_complex_example(self):
        """Test parsing with a complex real-world example."""
        text = """The paper titled "Model Guidance via Explanations Turns Image
        Classifiers into Segmentation Models" explores...

        Therefore, the paper's contributions are considered novel.

        **Label: 1**

        **References:**

        1. "Weakly Supervised Multi-Object Tracking and Segmentation"

        2. "Consistent Explanations by Contrastive Learning"

        3. "Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping"""

        result = parse_result(text)
        assert result.label == 1
        assert "Model Guidance via Explanations" in result.rationale
        assert "contributions are considered novel" in result.rationale
        assert "**References:**" in result.rationale

    def test_parse_result_label_in_middle_of_text(self):
        """Test parsing when label appears in the middle of text."""
        text = """Some initial thoughts about the paper.

        Label: 0

        Additional comments after the label."""

        result = parse_result(text)
        assert result.label == 0
        assert "Some initial thoughts" in result.rationale
        assert "Additional comments" in result.rationale

    def test_parse_result_case_insensitive(self):
        """Test that label parsing is case-insensitive."""
        text = """The paper is novel.

        LABEL: 1"""

        result = parse_result(text)
        assert result.label == 1

    def test_parse_result_no_label(self):
        """Test parsing when no label is found."""
        text = """The paper is interesting but there's no label here."""

        result = parse_result(text)
        assert result.label == 0  # GPTFull.error() returns label=0
        assert result.rationale == "<error>"
        assert not result.is_valid()

    def test_parse_result_empty_text(self):
        """Test parsing with empty text."""
        result = parse_result("")
        assert result.label == 0
        assert result.rationale == "<error>"
        assert not result.is_valid()

    def test_parse_result_none_text(self):
        """Test parsing with None text."""
        result = parse_result(None)
        assert result.label == 0
        assert result.rationale == "<error>"
        assert not result.is_valid()

    def test_parse_result_invalid_label_value(self):
        """Test parsing when label has invalid value."""
        text = """The paper is novel.

        Label: 2"""  # Invalid - should be 0 or 1

        result = parse_result(text)
        assert result.label == 0
        assert result.rationale == "<error>"
        assert not result.is_valid()

    def test_parse_result_label_after_approval_decision(self):
        """Test parsing when label appears in the approval decision sentence."""
        text = """The paper titled "ELEGANT: Certified Defense on the Fairness of Graph Neural Networks" addresses the critical issue of ensuring fairness in Graph Neural Networks (GNNs) under adversarial conditions.

        Given the approval decision is "False," the novelty label is 0.

        **References:**

        1. AGNNCert: Defending Graph Neural Networks against Arbitrary Perturbations with Deterministic Certification

        2. A Simple and Yet Fairly Effective Defense for Graph Neural Networks

        **Label:** 0"""

        result = parse_result(text)
        assert result.label == 0
        assert "ELEGANT" in result.rationale
        assert "Given the approval decision" in result.rationale
        assert "**References:**" in result.rationale
