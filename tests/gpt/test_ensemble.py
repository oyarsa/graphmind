"""Tests for ensemble aggregation helpers."""

from paper.gpt.ensemble import select_best_rationale_tfidf


def test_select_best_rationale_tfidf_prefers_non_empty_when_identical() -> None:
    """If all non-empty rationales are identical, return the first non-empty value."""
    best = select_best_rationale_tfidf(["", "same rationale", "same rationale"])
    assert best == "same rationale"


def test_select_best_rationale_tfidf_returns_empty_for_all_empty() -> None:
    """All-empty inputs should return an empty string."""
    best = select_best_rationale_tfidf(["", "   ", ""])
    assert best == ""
