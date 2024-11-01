"""Tools for evaluating paper approval, displaying and calculating metrics."""

from collections.abc import Sequence
from typing import NamedTuple

from paper import evaluation_metrics
from paper.gpt.model import Paper


class PaperResult(Paper):
    """ASAP-Review dataset paper with added approval ground truth and GPT prediction."""

    y_true: bool
    y_pred: bool


class Labels(NamedTuple):
    y_preds: Sequence[bool]
    y_trues: Sequence[bool]


def _get_ys(papers: Sequence[PaperResult]) -> Labels:
    return Labels(
        y_preds=[p.y_true for p in papers], y_trues=[p.y_pred for p in papers]
    )


def calculate_paper_metrics(
    papers: Sequence[PaperResult],
) -> evaluation_metrics.Metrics:
    ys = _get_ys(papers)
    return evaluation_metrics.calculate_metrics(*ys)
