"""Ensemble evaluation functions for aggregating multiple LLM evaluations.

This module provides functions for aggregating multiple evaluation rounds using majority
voting and rationale selection based on TF-IDF scoring.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from sklearn.feature_extraction.text import (  # pyright: ignore[reportMissingTypeStubs]
    TfidfVectorizer,
)

from paper.gpt.evaluate_paper import (
    GPTFull,
    GPTFullWithConfidence,
    GPTStructured,
    GPTStructuredRaw,
)

# This needs to be a TypeAlias as `type` can't be used for isinstance checks.
GPTPreConfidence: TypeAlias = GPTFull | GPTStructuredRaw  # noqa: UP040
GPTWithConfidence: TypeAlias = GPTFullWithConfidence | GPTStructured  # noqa: UP040


def select_best_rationale_tfidf(rationales: Sequence[str]) -> str:
    """Select the best rationale using a simple TF-IDF sum score.

    Falls back to the longest by word count if TF-IDF isn't available or if all non-empty
    rationales are identical.

    Args:
        rationales: Sequence of rationale strings.

    Returns:
        One rationale (empty string if all inputs are empty). The rationale is returned
        as-is.
    """
    if not rationales:
        return ""

    # Trim whitespace but keep original indices for returning the original string
    norm = [r.strip() for r in rationales]
    non_empty = [i for i, r in enumerate(norm) if r]

    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return rationales[non_empty[0]]

    # If everything is literally the same text, just pick the first
    if len({norm[i] for i in non_empty}) == 1:
        return rationales[0]

    try:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000,
        )
        scores = vec.fit_transform(norm)  # type: ignore[misc]

        # If no features (e.g., only stopwords), fall back
        if scores.shape[1] == 0:  # type: ignore[misc]
            return longest_by_word_count(rationales)

        scores = np.asarray(scores.sum(axis=1)).ravel()  # type: ignore[misc]
        best_idx = int(scores.argmax())
        return rationales[best_idx]
    except Exception:
        return longest_by_word_count(rationales)


def longest_by_word_count(items: Sequence[str]) -> str:
    """Pick the item with the highest word count; on ties, last wins."""

    def count_words(s: str) -> int:
        return len(re.findall(r"\w+", s))

    best_idx = max(range(len(items)), key=lambda i: count_words(items[i]))
    return items[best_idx]


def aggregate_ensemble_evaluations(
    evaluations: Sequence[GPTPreConfidence],
) -> GPTWithConfidence:
    """Aggregate multiple GPTFull evaluations using majority voting and rationale selection.

    Args:
        evaluations: List of valid GPTFull evaluations.

    Returns:
        Object with majority label, best rationale and confidence. See `GPTPreConfidence`
        and `GPTWithConfidence`.

    Raises:
        ValueError: If evaluations list is empty.
    """
    if not evaluations:
        raise ValueError("Cannot aggregate empty list of evaluations")

    if len(evaluations) == 1:
        return evaluations[0].with_confidence(1.0)

    label_counts = Counter(eval_.label for eval_ in evaluations)

    # Determine winning label (ties go to negative/0)
    if label_counts[1] > label_counts[0]:
        winning_label = 1
    else:
        winning_label = 0

    confidence = label_counts[winning_label] / len(evaluations)

    # Get evaluations with the winning label
    winning_evaluations = [
        eval_ for eval_ in evaluations if eval_.label == winning_label
    ]

    # Select best rationale using TF-IDF
    best_rationale = select_best_rationale_tfidf([
        eval_.rationale for eval_ in winning_evaluations
    ])
    # Find the evaluation that produced the best rationale
    best_evaluation = next(
        eval_ for eval_ in winning_evaluations if eval_.rationale == best_rationale
    )

    return best_evaluation.with_confidence(confidence)
