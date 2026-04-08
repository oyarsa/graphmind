"""Text similarity metrics using untyped third-party libraries (nltk, rouge_score, bert_score)."""

# pyright: basic

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def compute_rouge(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores averaged over pairs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(float(result[key].fmeasure))

    return {key: mean(values) for key, values in scores.items()}


def compute_bertscore(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute BERTScore F1 averaged over pairs."""
    import torch
    from bert_score import score

    _, _, f1 = score(
        list(predictions),
        list(references),
        lang="en",
        verbose=False,
    )
    assert isinstance(f1, torch.Tensor)
    return float(f1.mean().item())
