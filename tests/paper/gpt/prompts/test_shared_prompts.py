"""Tests for Python-based prompt templates."""

from paper.gpt.prompts import PromptTemplate
from paper.gpt.prompts._shared import (
    CONSERVATIVE,
    EVAL_SCALE,
    EVAL_SCALE_STRUCTURED,
    GRAPH_INTRO,
    NOVELTY_5,
    RATIONALE_BASIC,
    RATIONALE_NO_RELATED,
    RATIONALE_STRUCTURED,
    RELATED_INTRO,
    RELATED_WITH_IDS,
)
from paper.gpt.prompts.evaluate_graph import GRAPH_EVAL_USER_PROMPTS


class TestSharedConstants:
    """Test shared prompt constants."""

    def test_novelty_5_contains_scale(self) -> None:
        """Test that NOVELTY_5 contains the 1-5 rating scale."""
        assert "1 = Not novel" in NOVELTY_5
        assert "5 = Significant new problem" in NOVELTY_5

    def test_eval_scale_includes_novelty(self) -> None:
        """Test that EVAL_SCALE includes NOVELTY_5 via f-string composition."""
        assert "1 = Not novel" in EVAL_SCALE
        assert CONSERVATIVE in EVAL_SCALE
        assert RATIONALE_BASIC in EVAL_SCALE

    def test_rationale_structured_includes_novelty(self) -> None:
        """Test that RATIONALE_STRUCTURED includes NOVELTY_5."""
        assert "1 = Not novel" in RATIONALE_STRUCTURED

    def test_rationale_no_related_includes_novelty(self) -> None:
        """Test that RATIONALE_NO_RELATED includes NOVELTY_5."""
        assert "1 = Not novel" in RATIONALE_NO_RELATED

    def test_eval_scale_structured_includes_rationale(self) -> None:
        """Test that EVAL_SCALE_STRUCTURED includes RATIONALE_STRUCTURED."""
        assert "evidence from related papers" in EVAL_SCALE_STRUCTURED

    def test_graph_intro_describes_summary(self) -> None:
        """Test that GRAPH_INTRO describes the paper summary."""
        assert "paper summary" in GRAPH_INTRO.lower()

    def test_related_intro_describes_split(self) -> None:
        """Test that RELATED_INTRO describes supporting/contrasting split."""
        assert "supporting" in RELATED_INTRO
        assert "contrasting" in RELATED_INTRO

    def test_related_with_ids_mentions_ids(self) -> None:
        """Test that RELATED_WITH_IDS mentions paper IDs."""
        assert "ID" in RELATED_WITH_IDS


class TestPromptTemplates:
    """Test PromptTemplate instances from evaluate_graph module."""

    def test_full_graph_structured_exists(self) -> None:
        """Test that full-graph-structured prompt exists."""
        assert "full-graph-structured" in GRAPH_EVAL_USER_PROMPTS

    def test_sans_prompt_exists(self) -> None:
        """Test that sans prompt exists."""
        assert "sans" in GRAPH_EVAL_USER_PROMPTS

    def test_related_prompt_exists(self) -> None:
        """Test that related prompt exists."""
        assert "related" in GRAPH_EVAL_USER_PROMPTS

    def test_norel_graph_prompt_exists(self) -> None:
        """Test that norel-graph prompt exists."""
        assert "norel-graph" in GRAPH_EVAL_USER_PROMPTS

    def test_prompts_have_template_vars(self) -> None:
        """Test that prompts have runtime template variables."""
        full = GRAPH_EVAL_USER_PROMPTS["full-graph-structured"]
        assert "{title}" in full.template
        assert "{abstract}" in full.template
        assert "{demonstrations}" in full.template

    def test_prompts_have_rating_scale(self) -> None:
        """Test that prompts contain the rating scale."""
        for name in ["full-graph-structured", "sans", "related", "norel-graph"]:
            prompt = GRAPH_EVAL_USER_PROMPTS[name]
            assert "1 = Not novel" in prompt.template or "1 =" in prompt.template, (
                f"Missing rating scale in {name}"
            )

    def test_prompt_template_format(self) -> None:
        """Test that PromptTemplate.format works correctly."""
        prompt = PromptTemplate(
            name="test",
            template="Title: {title}\nAbstract: {abstract}",
            system="Test system",
            type_name="TestType",
        )
        result = prompt.format(title="My Title", abstract="My Abstract")
        assert result == "Title: My Title\nAbstract: My Abstract"

    def test_prompt_template_with_user(self) -> None:
        """Test that PromptTemplate.with_user creates a Prompt instance."""
        prompt = PromptTemplate(
            name="test",
            template="Hello",
            system="System prompt",
            type_name="TestType",
        )
        result = prompt.with_user("User message")
        assert result.system == "System prompt"
        assert result.user == "User message"

    def test_all_prompts_have_type_names(self) -> None:
        """Test that all prompts have type_name set."""
        for name, prompt in GRAPH_EVAL_USER_PROMPTS.items():
            assert prompt.type_name, f"Prompt {name} has no type_name"

    def test_no_unresolved_shared_refs(self) -> None:
        """Test that no prompts contain unresolved {shared.x.y} references."""
        for name, prompt in GRAPH_EVAL_USER_PROMPTS.items():
            assert "{shared." not in prompt.template, (
                f"Unresolved shared ref in {name} template"
            )
            assert "{shared." not in prompt.system, (
                f"Unresolved shared ref in {name} system"
            )
