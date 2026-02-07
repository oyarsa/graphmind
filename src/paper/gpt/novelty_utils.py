"""Utilities for extracting and analysing novelty labels."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from paper.gpt.run_gpt import GPTResult, gpt_is_valid, gpt_sequence
from paper.types import Immutable
from paper.util import clamp

if TYPE_CHECKING:
    from paper.gpt.evaluate_paper import GPTStructuredRaw
    from paper.gpt.run_gpt import LLMClient

logger = logging.getLogger(__name__)


class NoveltyResult(Immutable):
    """Result of evaluating novelty given paper information and external evidence."""

    rating: Annotated[
        int,
        Field(
            description="Novelty rating where 1 is not novel and 4 is very novel.",
            ge=1,
            le=4,
        ),
    ]


def _is_not_novel(label: int) -> bool:
    """Return whether a 1-5 novelty label should be treated as not novel."""
    return label < 4


async def get_novelty_probability(
    client: LLMClient,
    output: GPTResult[GPTStructuredRaw],
    best_of_n: int = 0,
) -> GPTResult[float]:
    """Get novelty probability from evaluation output.

    We re-prompt the LLM to give N ratings and calculate the probability as the
    percentage.

    If set to 0, returns 0 for not-novel labels (1-3) and 1 for novel labels (4-5).

    Args:
        client: LLM client to use to generate best of N results.
        output: Evaluation output. Used as input for best of N results.
        best_of_n: How many ratings we should sample. Defaults to 0.

    Returns:
        Probability as a float between 0 and 1. A value of 0.8 means 80% probability
        the paper is novel.
    """
    if best_of_n == 0:
        return output.map(lambda out: 0.0 if _is_not_novel(out.label) else 1.0)

    prob = await output.abind(lambda out: get_novelty_best_of_n(client, out, best_of_n))

    # "not novel" will have at most 40% prob, and "novel" will have at least 60%
    return prob.map(
        lambda p: min(p, 0.4) if _is_not_novel(output.result.label) else max(p, 0.6)
    )


BEST_OF_SYSTEM_PROMPT = """You are an expert reviewer assessing a paper's novelty on a
scale from 1 to 4. You are given an expert assessment of the paper containing a summary
and key evidence arguing for and against the novelty. Your task is to translate this
assessment into a rating.

Rating scale:
1 - Not novel: Use this when contradictory evidence outweighs supporting evidence
2 - Somewhat novel: Use this only when evidence is truly balanced
3 - Novel: Use this when supporting evidence outweighs contradictory evidence
4 - Very novel: Use this when supporting evidence is strong with minimal concerns

IMPORTANT DISTRIBUTION GUIDANCE:
- For "not novel" papers: most papers should be rated 1.
- For "novel" papers: most papers should be rated 4.
- Ratings 2 and 3 are for genuinely ambiguous cases, not default choices

Look at the balance of evidence. If one side clearly dominates, use the extreme
rating (1 or 4). Only use middle ratings when the evidence is genuinely mixed.
"""

BEST_OF_USER_TEMPLATE = """As an independent reviewer, rate this paper's
novelty on a scale from 1 to 4 based on the following analysis.

Analysis and evidence:
{rationale}

Initial assessment: {label_text}

This initial assessment suggests the paper leans toward being {label_text}.

Based on your independent assessment, what rating from 1 to 4 best reflects
this paper's novelty?"""

_RNG = random.Random(0)


async def get_novelty_best_of_n(
    client: LLMClient, output: GPTStructuredRaw, n: int
) -> GPTResult[float]:
    """Get novelty probability from re-prompting the LLM for best of N results.

    We use another round of calls to the LLM with a custom prompt to ask it to review
    the evidence highlighted by the evaluation result, and generate a novelty rating
    from 1-4. The ratings are then averaged to produce a probability.

    Args:
        client: LLM client to use to generate best of N results.
        output: Evaluation output. Used as input for best of N results.
        n: Number of results to prompt for.

    Returns:
        Probability as a float between 0 and 1, calculated as the average of all
        ratings divided by 4. For example, ratings [2, 3, 3, 4, 3] would yield
        (2+3+3+4+3)/(5*4) = 15/20 = 0.75.
    """
    tasks = [
        client.run(
            NoveltyResult,
            BEST_OF_SYSTEM_PROMPT,
            BEST_OF_USER_TEMPLATE.format(
                rationale=output.rationale,
                label_text="not novel" if _is_not_novel(output.label) else "novel",
            ),
            temperature=1,
            seed=_RNG.randint(1, 100),
        )
        for _ in range(n)
    ]
    task_results = await asyncio.gather(*tasks)
    valid_results = gpt_sequence(r for r in task_results if gpt_is_valid(r))
    if not valid_results.result:
        fallback = 0.0 if _is_not_novel(output.label) else 1.0
        logger.debug(
            "No valid best-of-N results. Falling back to label bucket probability: %.1f",
            fallback,
        )
        return valid_results.map(lambda _: fallback)

    return valid_results.map(
        lambda results: sum(clamp(r.rating, 1, 4) for r in results) / (4 * len(results))
    )
