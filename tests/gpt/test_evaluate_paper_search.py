"""Tests for the evaluate_paper_search module."""

from textwrap import dedent
from paper.gpt.evaluate_paper_search import parse_result


class TestParseResult:
    """Test the parse_result function with various output formats."""

    def test_parse_result_with_bold_label(self):
        """Test parsing with **Label: 1** format."""
        text = """The paper is novel.

        **Label: 1**

        **References:**
        1. Some paper"""

        result = parse_result(dedent(text))
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

        result = parse_result(dedent(text))
        assert result.label == 0
        assert "The paper is not novel." in result.rationale

    def test_parse_result_with_plain_label(self):
        """Test parsing with plain Label: 1 format."""
        text = """The paper is novel.

        Label: 1

        References:
        1. Some paper"""

        result = parse_result(dedent(text))
        assert result.label == 1
        assert "The paper is novel." in result.rationale

    def test_parse_result_with_rationale_prefix(self):
        """Test parsing with Rationale: prefix."""
        text = """Rationale: The paper introduces novel concepts.

        Label: 1"""

        result = parse_result(dedent(text))
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

        result = parse_result(dedent(text))
        assert result.label == 1
        assert "Model Guidance via Explanations" in result.rationale
        assert "contributions are considered novel" in result.rationale
        assert "**References:**" in result.rationale

    def test_parse_result_label_in_middle_of_text(self):
        """Test parsing when label appears in the middle of text."""
        text = """Some initial thoughts about the paper.

        Label: 0

        Additional comments after the label."""

        result = parse_result(dedent(text))
        assert result.label == 0
        assert "Some initial thoughts" in result.rationale
        assert "Additional comments" in result.rationale

    def test_parse_result_case_insensitive(self):
        """Test that label parsing is case-insensitive."""
        text = """The paper is novel.

        LABEL: 1"""

        result = parse_result(dedent(text))
        assert result.label == 1

    def test_parse_result_no_label(self):
        """Test parsing when no label is found."""
        text = """The paper is interesting but there's no label here."""

        result = parse_result(dedent(text))
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

        result = parse_result(dedent(text))
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

        result = parse_result(dedent(text))
        assert result.label == 0
        assert "ELEGANT" in result.rationale
        assert "Given the approval decision" in result.rationale
        assert "**References:**" in result.rationale

    def test_parse_result_no_label_format(self):
        """Test parsing when text doesn't follow the expected format at all.

        This test documents a known failure case where the LLM output doesn't include
        a properly formatted "Label: <0 or 1>" line, instead embedding the label value
        in the narrative text.
        """
        text = """The paper titled "LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving" explores the integration of Large Language Models (LLMs) into autonomous driving systems to enhance decision-making capabilities, particularly in complex scenarios requiring human-like commonsense reasoning. The authors propose cognitive pathways for comprehensive reasoning with LLMs and develop algorithms to translate LLM decisions into actionable driving commands, aiming to improve safety, efficiency, generalizability, and interpretability in autonomous driving.

        Upon reviewing recent literature, several studies have investigated similar applications of LLMs in autonomous driving:

        1. "Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving" (December 2024) integrates an LLM-based driving expert into deep reinforcement learning to enhance decision-making efficiency and performance in autonomous vehicles.

        2. "Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving" (February 2024) examines the adaptation of LLMs for arithmetic and commonsense reasoning in dynamic driving situations, focusing on analyzing sensor data and understanding driving regulations.

        These studies collectively demonstrate a growing interest in employing LLMs to enhance decision-making in autonomous driving, focusing on integrating human-like reasoning, improving interpretability, and handling complex driving scenarios. The "LanguageMPC" paper contributes to this evolving field by proposing specific cognitive pathways and algorithms for translating LLM decisions into driving commands. However, given the substantial overlap with existing research, particularly in integrating LLMs for decision-making and enhancing interpretability in autonomous driving, the novelty of the "LanguageMPC" paper is limited.

        Therefore, the novelty label for this paper is 0.

        **References:**

        1. "Large Language Model guided Deep Reinforcement Learning for Decision Making in Autonomous Driving" (https://arxiv.org/abs/2412.18511)

        2. "Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving" (https://arxiv.org/abs/2402.13602)"""

        result = parse_result(dedent(text))
        # This should fail because there's no "Label: 0" or "Label: 1" line
        assert result.label == 0  # GPTFull.error() returns label=0
        assert result.rationale == "<error>"
        assert not result.is_valid()
