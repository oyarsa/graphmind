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


def find_novel_field_token(logprobs: list[TokenProb]) -> TokenProb | None:
    """Find the novel field token by looking for 'novel' + '": ' pattern.

    Based on tokenization analysis, the consistent pattern is:
    'novel' + '": ' + (optional space) + 'yes' or 'no'

    Args:
        logprobs: List of token probabilities from GPT output.

    Returns:
        Tuple of (index, TokenProb) for the novel value, or None if not found.
    """
    if not logprobs:
        return None

    # Look for the consistent pattern: 'nov' + 'el' + '":"' + yes/no
    for i in range(len(logprobs) - 4):
        if (
            logprobs[i].token == "nov"  # noqa: S105
            and i + 1 < len(logprobs)
            and logprobs[i + 1].token == "el"  # noqa: S105
            and i + 2 < len(logprobs)
            and logprobs[i + 2].token == '":"'  # noqa: S105
            and i + 3 < len(logprobs)
        ):
            next_token = logprobs[i + 3]
            if next_token.token in ["yes", "no"]:
                return next_token

            # Check if there's a space, then the yes/no
            if (
                next_token.token == " "  # noqa: S105
                and i + 4 < len(logprobs)
                and logprobs[i + 4].token in ["yes", "no"]
            ):
                return logprobs[i + 4]

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

    # Calculate confidence
    if positive_token in probs and negative_token in probs:
        # Both options present, calculate relative probability
        return probs[positive_token] / (probs[positive_token] + probs[negative_token])
    elif positive_token in probs:
        # Only found positive token, interpret probability directly
        return probs[positive_token]
    elif negative_token in probs:
        # Only found negative token, confidence in positive is inverse
        return 1 - probs[negative_token]
    else:
        # Neither token found
        return 0


def calculate_novel_confidence(novel_token: TokenProb) -> float:
    """Calculate novel confidence from novel field token probabilities.

    This calculates the probability that the paper is novel (novel=yes) vs not novel
    (novel=no) based on the token probabilities. It considers both the chosen token and
    alternatives.

    Args:
        novel_token: TokenProb for the novel field position, with potential alternatives.

    Returns:
        Confidence as a float between 0 and 1, or 0 if can't calculate.
        A value of 0.8 means 80% confidence the paper is novel.
    """
    return calculate_binary_confidence(novel_token, "yes", "no")


def calculate_label_confidence(label_token: TokenProb) -> float:
    """Calculate novelty confidence from label token probabilities.

    This calculates the probability that the paper is novel (label=1) vs not novel
    (label=0) based on the token probabilities. It considers both the chosen token and
    alternatives.

    Args:
        label_token: TokenProb for the label position, with potential alternatives.

    Returns:
        Confidence as a float between 0 and 1, or None if can't calculate.
        A value of 0.8 means 80% confidence the paper is novel.
    """
    return calculate_binary_confidence(label_token, "1", "0")


def get_novelty_probability(
    logprobs: list[TokenProb] | None, format: NoveltyFormat
) -> float:
    """Get novelty probability directly from GPTResult logprobs.

    This is a convenience function that finds the appropriate token and calculates
    the probability that the paper is novel in one step.

    Args:
        logprobs: The logprobs list from GPTResult.logprobs.
        format: Whether to extract from "label" (0/1) or "novel" (yes/no) field.

    Returns:
        Probability as a float between 0 and 1. Returns 0.5 if can't calculate.
        A value of 0.8 means 80% probability the paper is novel.
    """
    if not logprobs:
        logger.debug("Cannot calculate probability: null or empty logprobs")
        return 0.5  # Default to neutral if no logprobs

    match format:
        case NoveltyFormat.LABEL:
            token_info = find_novelty_label_token(logprobs)
            if token_info is None:
                return 0.5  # Default to neutral if can't find token
            return calculate_label_confidence(token_info)
        case NoveltyFormat.WORD:
            token_info = find_novel_field_token(logprobs)
            if token_info is None:
                return 0.5  # Default to neutral if can't find token
            return calculate_novel_confidence(token_info)


def best_novelty_probability(logprobs: list[TokenProb] | None) -> float:
    """Use the least extreme probability between the label and novel fields.

    We calculate the probabilities from both "label" (0/1) and "novel" ("yes"/"no")
    and use the less extreme one.
    """
    label_confidence = get_novelty_probability(logprobs, NoveltyFormat.LABEL)
    word_confidence = get_novelty_probability(logprobs, NoveltyFormat.WORD)
    return least_extreme(label_confidence, word_confidence)


def least_extreme(x: float, y: float) -> float:
    """Assuming x and y are in 0..1, return the less extreme (furthest from 0 and 1)."""
    assert 0 <= x <= 1
    assert 0 <= y <= 1
    return x if abs(x - 0.5) <= abs(y - 0.5) else y
