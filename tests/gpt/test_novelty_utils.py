"""Tests for novelty probability helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest
from pydantic import ValidationError

from paper.gpt.evaluate_paper import GPTStructuredRaw
from paper.gpt.novelty_utils import NoveltyResult, get_novelty_probability
from paper.gpt.result import GPTResult
from paper.gpt.run_gpt import LLMClient


def _structured(label: int) -> GPTStructuredRaw:
    return GPTStructuredRaw(
        paper_summary="summary",
        supporting_evidence=[],
        contradictory_evidence=[],
        key_comparisons=[],
        conclusion="conclusion",
        label=label,
    )


@dataclass
class _StubClient:
    ratings: Sequence[int]
    user_prompts: list[str] = field(default_factory=list[str])

    async def run(
        self,
        class_: type[NoveltyResult],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[NoveltyResult]:
        del class_, system_prompt, max_tokens, temperature, seed
        self.user_prompts.append(user_prompt)
        return GPTResult(result=NoveltyResult(rating=next(iter(self.ratings))), cost=0)


def test_novelty_result_requires_rating() -> None:
    """NoveltyResult must not accept missing rating."""
    with pytest.raises(ValidationError):
        NoveltyResult.model_validate({})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (1, 0.0),
        (2, 0.0),
        (3, 0.0),
        (4, 1.0),
        (5, 1.0),
    ],
)
async def test_probability_without_best_of_n_maps_1_to_5_labels(
    label: int, expected: float
) -> None:
    """best_of_n=0 should map labels 1-3 to not-novel and 4-5 to novel."""
    output = GPTResult(result=_structured(label), cost=0)
    probability = await get_novelty_probability(
        cast(LLMClient, _StubClient([4])), output, best_of_n=0
    )
    assert probability.result == expected


@pytest.mark.asyncio
async def test_probability_clamping_uses_not_novel_bucket_for_labels_below_4() -> None:
    """Low labels should be capped at 0.4 after best-of-N sampling."""
    output = GPTResult(result=_structured(1), cost=0)
    # Raw best-of-N would be 1.0, but not-novel labels should cap at 0.4.
    probability = await get_novelty_probability(
        cast(LLMClient, _StubClient([4, 4, 4])), output, best_of_n=3
    )
    assert probability.result == 0.4


@pytest.mark.asyncio
async def test_probability_clamping_uses_novel_bucket_for_labels_4_and_5() -> None:
    """High labels should floor at 0.6 after best-of-N sampling."""
    output = GPTResult(result=_structured(5), cost=0)
    # Raw best-of-N would be 0.25, but novel labels should floor at 0.6.
    probability = await get_novelty_probability(
        cast(LLMClient, _StubClient([1, 1, 1])), output, best_of_n=3
    )
    assert probability.result == 0.6


@pytest.mark.asyncio
async def test_best_of_n_prompt_uses_label_bucket_text() -> None:
    """Prompt should say not novel for labels 1-3, novel for 4-5."""
    low_client = _StubClient([2])
    high_client = _StubClient([2])

    await get_novelty_probability(
        cast(LLMClient, low_client),
        GPTResult(result=_structured(3), cost=0),
        best_of_n=1,
    )
    await get_novelty_probability(
        cast(LLMClient, high_client),
        GPTResult(result=_structured(4), cost=0),
        best_of_n=1,
    )

    assert "leans toward being not novel" in low_client.user_prompts[0]
    assert "leans toward being novel" in high_client.user_prompts[0]
