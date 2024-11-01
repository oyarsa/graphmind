"""Tools for evaluating paper approval, displaying and calculating metrics."""

from collections.abc import Sequence
from typing import NamedTuple

from pydantic import Field

from paper import evaluation_metrics
from paper.gpt.model import Paper
from paper.util import safediv


class PaperResult(Paper):
    """ASAP-Review dataset paper with added approval ground truth and GPT prediction."""

    y_true: bool = Field(description="Human annotation")
    y_pred: bool = Field(description="Model prediction")
    rationale: str = Field(description="Model rationale for the prediction")


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
    """Calculate classification metrics between `y_true` and `y_pred` results.

    See also `paper.evaluation_metrics.calculate_metrics`.
    """
    return evaluation_metrics.calculate_metrics(*_get_ys(papers))


def display_metrics(
    metrics: evaluation_metrics.Metrics, results: Sequence[PaperResult]
) -> str:
    """Display metrics and distribution statistics from the results.

    `evaluation_metrics.Metrics` are displayed directly. The distribution statistics are
    shown as the count and percentage of true/false for both gold and prediction.

    Args:
        metrics: Metrics calculated using `evaluation_metrics.calculate_metrics`.
        results: Paper evaluation results.

    Returns:
        Formatted string showing both metrics and distribution statistics.
    """
    y_true = [r.y_true for r in results]
    y_pred = [r.y_pred for r in results]

    output = [
        "Metrics:",
        str(metrics),
        "",
        f"Gold (P/N): {sum(y_true)}/{len(y_true) - sum(y_true)}"
        f" ({safediv(sum(y_true), len(y_true)):.2%})",
        f"Pred (P/N): {sum(y_pred)}/{len(y_pred) - sum(y_pred)}"
        f" ({safediv(sum(y_pred), len(y_pred)):.2%})",
    ]
    return "\n".join(output)
