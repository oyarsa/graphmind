"""Utilities for extracting and analyzing novelty labels from GPT logprobs."""

# TODO: Vibe-coded. Review.

from __future__ import annotations

import logging
from enum import Enum
from math import exp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper.gpt.run_gpt import TokenProb

logger = logging.getLogger(__name__)


class NoveltyFormat(Enum):
    """Format of novelty output to extract probability from."""

    LABEL = "label"  # Extract from "label": 0/1 field
    WORD = "word"  # Extract from "novel": yes/no field


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


def get_novelty_probability(logprobs: list[TokenProb] | None) -> float:
    """Get novelty probability directly from GPTResult logprobs.

    This is a convenience function that finds the appropriate token and calculates
    the probability that the paper is novel in one step.

    We first find the token respective to the "label" number and calculate the
    probability from the top values.

    Args:
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
