"""Tools for evaluating paper novelty, displaying and calculating metrics."""

from collections.abc import Sequence
from typing import Annotated, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from paper import evaluation_metrics
from paper.gpt.model import Paper
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.util import safediv


class PaperResult(Paper):
    """PeerRead paper with added novelty rating ground truth and GPT prediction."""

    y_true: Annotated[int, Field(description="Human annotation")]
    y_pred: Annotated[int, Field(description="Model prediction")]
    rationale_pred: Annotated[
        str, Field(description="Model rationale for the prediction")
    ]


class Labels(NamedTuple):
    """Prediction and ground truth labels."""

    y_preds: Sequence[int]
    y_trues: Sequence[int]


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


EVALUATE_DEMONSTRATION_PROMPTS = load_prompts("eval_demonstrations")


class Demonstration(BaseModel):
    """Paper for evaluation demos with full information and demonstration type."""

    model_config = ConfigDict(frozen=True)

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Paper abstract")]
    text: Annotated[str, Field(description="Paper full main text")]
    rationale: Annotated[str, Field(description="Rationale given by a reviewer")]
    rating: Annotated[int, Field(description="Rating from the rationale")]


def format_demonstrations(
    demonstrations: Sequence[Demonstration], prompt: PromptTemplate
) -> str:
    """Format all `demonstrations` according to `prompt` as a single string.

    If `demonstrations` is empty, returns the empty string.
    """
    if not demonstrations:
        return ""

    output_all = [
        "-Demonstrations-\n"
        "The following are examples of other paper evaluations with their novelty"
        " ratings and rationales:\n",
    ]

    output_all.extend(
        prompt.template.format(
            title=demo.title,
            abstract=demo.abstract,
            main_text=demo.text,
            rationale=demo.rationale,
        )
        for demo in demonstrations
    )
    return f"\n{"-" * 50}\n".join(output_all)


class GPTFull(BaseModel):
    """Decision on if the paper should be published and the reason for the decision."""

    model_config = ConfigDict(frozen=True)

    rationale: Annotated[str, Field(description="How you reached your novelty rating.")]
    rating: Annotated[int, Field(description="How novel the paper is judged to be.")]


CLASSIFY_TYPES = {
    "full": GPTFull,
}
