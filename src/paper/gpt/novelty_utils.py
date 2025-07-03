"""Utilities for extracting and analyzing novelty labels from GPT logprobs."""

from __future__ import annotations

import asyncio
import logging
import os
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
        int, Field("Novelty rating when 0 is not novel at all and 5 is very novel.")
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
        - "logprob": use token logprobs
        - N (int): parameter for best of N method
        - <unset>: returns 0 for "not novel" and 1 for "novel"

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

    # return prob.map(lambda p: min(p, 0.5) if output.result.label == 0 else max(p, 0.5))
    return prob


BEST_OF_SYSTEM_PROMPT = """You are one of several expert reviewers independently
assessing a paper's novelty on a scale from 1 to 5.

Rating scale:
1 - Not novel: Ideas clearly exist in prior work with minimal changes
2 - Slightly novel: Minor variations or incremental improvements on existing work
3 - Moderately novel: Notable advances with some original elements
4 - Quite novel: Significant new contributions with substantial originality
5 - Highly novel: Groundbreaking ideas that represent major advances

Consider factors like:
- The significance of the core contribution
- How much the work differs from prior art
- Whether similar ideas exist but in different contexts
- The potential impact of the proposed approach

Different reviewers may reasonably assign different ratings based on how they
weigh these factors. Make your own independent judgment."""

BEST_OF_USER_TEMPLATE = """As an independent reviewer, rate this paper's
novelty on a scale from 1 to 5 based on the following analysis.

Analysis and evidence:
{rationale}

Different reviewers may weigh the supporting and contradictory evidence
differently, leading to different ratings. Some may focus more on technical
advances, others on conceptual novelty, and others on practical impact.

Based on your independent assessment, what rating from 1 to 5 best reflects
this paper's novelty?"""


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
            BEST_OF_USER_TEMPLATE.format(rationale=output.rationale),
            temperature=1,
        )
        for _ in range(n)
    ]
    task_results = await asyncio.gather(*tasks)
    valid_results = gpt_sequence(r for r in task_results if gpt_is_valid(r))
    return valid_results.map(
        lambda results: sum(clamp(r.rating, 1, 5) for r in results) / (5 * len(results))
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
