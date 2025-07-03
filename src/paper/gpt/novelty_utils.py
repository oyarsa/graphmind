"""Utilities for extracting and analyzing novelty labels from GPT logprobs."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from math import exp
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from paper.gpt.run_gpt import GPTResult, gpt_is_valid, gpt_sequence
from paper.types import Immutable
from paper.util import clamp

if TYPE_CHECKING:
    from paper.gpt.evaluate_paper import GPTStructuredRaw
    from paper.gpt.run_gpt import LLMClient, TokenProb

logger = logging.getLogger(__name__)


def find_novelty_label_token(logprobs: list[TokenProb]) -> TokenProb | None:
    """Find the novelty label token by looking for 'label' + '":' pattern.

    Based on tokenization analysis, the consistent pattern is:
    'label' + '":' + (optional space) + '0' or '1'

    Args:
        logprobs: List of token probabilities from GPT output.

    Returns:
        TokenProb for the label value, or None if not found.
    """
    if not logprobs:
        return None

    # Look for the consistent pattern: 'label' + '":' + (optional space) + digit
    for i in range(len(logprobs) - 2):
        if (
            logprobs[i].token == "label"  # noqa: S105
            and i + 1 < len(logprobs)
            and logprobs[i + 1].token == '":'  # noqa: S105
            and i + 2 < len(logprobs)
        ):
            next_token = logprobs[i + 2]
            if next_token.token in ["0", "1"]:
                return next_token

            # Check if there's a space, then the digit
            if (
                next_token.token == " "  # noqa: S105
                and i + 3 < len(logprobs)
                and logprobs[i + 3].token in ["0", "1"]
            ):
                return logprobs[i + 3]

    return None


def calculate_binary_confidence(
    token: TokenProb, positive_token: str, negative_token: str
) -> float:
    """Calculate confidence between two tokens given the positive and negative variants.

    Args:
        token: TokenProb with the chosen token and alternatives.
        positive_token: Token that indicates a positive/novel result.
        negative_token: Token that indicates a negative/not-novel result.

    Returns:
        Confidence as a float between 0 and 1, or 0 if can't calculate.
        A value of 0.8 means 80% confidence in the positive result.
    """
    # Collect probabilities for both tokens
    probs: dict[str, float] = {}

    # Add the chosen token
    if token.token in [positive_token, negative_token]:
        probs[token.token] = exp(token.logprob)

    # Add alternatives if available
    if token.top_logprobs:
        for alt in token.top_logprobs:
            if alt.token in [positive_token, negative_token]:
                probs[alt.token] = exp(alt.logprob)

    match (probs.get(positive_token), probs.get(negative_token)):
        case (None, None):  # No tokens found
            return 0.5
        case (p_pos, None):  # Only positive found: use directly
            return p_pos
        case (None, p_neg):  # Only negative found: use complement
            return 1 - p_neg
        case (p_pos, p_neg):  # Both found: use relative probability
            return p_pos / (p_pos + p_neg)


class NoveltyResult(Immutable):
    """Result of evaluating novelty given paper information and external evidence."""

    rating: Annotated[
        int, Field("Novelty rating when 1 is not novel at all and 4 is very novel.")
    ]


async def get_novelty_probability(
    client: LLMClient,
    output: GPTResult[GPTStructuredRaw],
) -> GPTResult[float]:
    """Get novelty probability from evaluation output.

    Can either calculate it from the logprob, re-prompt the LLM to give N results and
    calculate the percentage, or just give 0/1 for novel/not novel.

    Args:
        client: LLM client to use to generate best of N results.
        output: Evaluation output. Used as input for best of N results.

    Environment variables:
        PROB_METHOD: What method to use to determine the probability:
        - N: parameter for best of N method
        - <unset>: best of N with N=5
        - 0: returns 0 for "not novel" and 1 for "novel"

    Returns:
        Probability as a float between 0 and 1. Returns 0.5 if can't calculate.
        A value of 0.8 means 80% probability the paper is novel.
    """
    match method := os.getenv("PROB_METHOD"):
        case "logprob":
            prob = output.map(
                lambda _: get_novelty_probability_logbprob(output.logprobs)
            )
        case str() if method.isdigit():
            prob = await output.abind(
                lambda out: get_novelty_best_of_n(client, out, int(method))
            )
        case _:
            prob = output.map(lambda out: 0.0 if out.label == 0 else 1.0)

    return prob.map(lambda p: min(p, 0.4) if output.result.label == 0 else max(p, 0.6))


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
    from 1-5. The ratings are then averaged to produce a probability.

    Args:
        client: LLM client to use to generate best of N results.
        output: Evaluation output. Used as input for best of N results.
        n: Number of results to prompt for.

    Returns:
        Probability as a float between 0 and 1, calculated as the average of all
        ratings divided by 5. For example, ratings [2, 3, 3, 4, 3] would yield
        (2+3+3+4+3)/(5*5) = 15/25 = 0.6.
    """
    tasks = [
        client.run(
            NoveltyResult,
            BEST_OF_SYSTEM_PROMPT,
            BEST_OF_USER_TEMPLATE.format(
                rationale=output.rationale,
                label_text="not novel" if output.label == 0 else "novel",
            ),
            temperature=1,
            seed=_RNG.randint(1, 100),
        )
        for _ in range(n)
    ]
    task_results = await asyncio.gather(*tasks)
    valid_results = gpt_sequence(r for r in task_results if gpt_is_valid(r))
    logger.warning(f"{[x.rating for x in valid_results.result]}")
    return valid_results.map(
        lambda results: sum(clamp(r.rating, 1, 4) for r in results) / (4 * len(results))
    )


def get_novelty_probability_logbprob(logprobs: list[TokenProb] | None) -> float:
    """Get novelty probability directly from GPTResult logprobs.

    This is a convenience function that finds the appropriate token and calculates
    the probability that the paper is novel in one step.

    We first find the token respective to the "label" number and calculate the
    probability from the top values.

    Args:
        output: Evaluation output. Used as input for best of N results.
        logprobs: The logprobs list from GPTResult.logprobs.

    Returns:
        Probability as a float between 0 and 1. Returns 0.5 if can't calculate.
        A value of 0.8 means 80% probability the paper is novel.
    """
    if not logprobs:
        logger.debug("Cannot calculate probability: null or empty logprobs")
        return 0.5  # Default to neutral if no logprobs

    token_info = find_novelty_label_token(logprobs)
    if token_info is None:
        return 0.5  # Default to neutral if can't find token
    return calculate_binary_confidence(token_info, "1", "0")
