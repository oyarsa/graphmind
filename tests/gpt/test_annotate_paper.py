"""Tests for annotation pipeline helpers."""

from __future__ import annotations

from typing import cast

import pytest

from paper.gpt.annotate_paper import (
    ABS_SYSTEM_PROMPT,
    GPTAbstractClassify,
    _annotate_paper_single,
)
from paper.gpt.model import PaperTermRelation, PaperTerms
from paper.gpt.prompts import PromptTemplate
from paper.gpt.result import GPTResult
from paper.gpt.run_gpt import LLMClient
from paper.semantic_scholar.model import Paper


class _StubClient:
    async def run(
        self,
        class_: type[PaperTerms | GPTAbstractClassify],
        system_prompt: str,
        user_prompt: str,
    ) -> GPTResult[PaperTerms | GPTAbstractClassify]:
        del system_prompt, user_prompt
        if class_ is PaperTerms:
            return GPTResult(
                result=PaperTerms(
                    tasks=["task"],
                    methods=["method"],
                    metrics=[],
                    resources=[],
                    relations=[PaperTermRelation(head="task", tail="method")],
                ),
                cost=1.5,
            )

        return GPTResult(
            result=GPTAbstractClassify(
                background="background sentence", target="target sentence"
            ),
            cost=2.5,
        )


@pytest.mark.asyncio
async def test_annotate_paper_single_uses_both_costs_and_consistent_prompt() -> None:
    """Annotation result should include both API costs and matching prompt metadata."""
    paper = Paper(
        paper_id="paper-1",
        corpus_id=None,
        url=None,
        title="A title",
        abstract="An abstract",
        year=2025,
        publication_date=None,
        reference_count=0,
        citation_count=0,
        influential_citation_count=0,
        tldr=None,
        authors=None,
        venue=None,
    )

    term_prompt = PromptTemplate(name="term", template="{title} :: {abstract}")
    abstract_prompt = PromptTemplate(
        name="abstract", template="{demonstrations}\n{abstract}"
    )

    result = await _annotate_paper_single(
        client=cast(LLMClient, _StubClient()),
        paper=paper,
        user_prompt_term=term_prompt,
        user_prompt_abstract=abstract_prompt,
        abstract_demonstrations="demo text",
    )

    assert result.cost == 4.0
    assert result.result.prompt.system == ABS_SYSTEM_PROMPT
