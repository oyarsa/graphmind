"""Tools for evaluating paper novelty, displaying and calculating metrics."""

import logging
import os
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Annotated, NamedTuple, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from paper import evaluation_metrics
from paper import semantic_scholar as s2
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.util import safediv
from paper.util.serde import load_data, replace_fields

logger = logging.getLogger(__name__)


class PaperResult(s2.PaperWithS2Refs):
    """PeerRead paper with added novelty rating ground truth and GPT prediction."""

    y_true: Annotated[int, Field(description="Human annotation")]
    y_pred: Annotated[int, Field(description="Model prediction")]
    rationale_true: Annotated[str, Field(description="Human rationale annotation")]
    rationale_pred: Annotated[
        str, Field(description="Model rationale for the prediction")
    ]

    @classmethod
    def from_s2peer(
        cls, paper: s2.PaperWithS2Refs, y_pred: int, rationale_pred: str
    ) -> Self:
        """Construct `PaperResult` from the original paper and model predictions."""
        return cls.model_validate(
            paper.model_dump()
            | {
                "y_true": rating_to_binary(paper.rating),
                "rationale_true": paper.rationale,
                "y_pred": rating_to_binary(y_pred),
                "rationale_pred": rationale_pred,
            }
        )


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

    output = ["Metrics:", str(metrics)]
    for values, label in [(y_true, "Gold"), (y_pred, "Predicted")]:
        output.append(f"\n{label} distribution:")
        for rating in range(1, 6):
            count = sum(y == rating for y in values)
            output.append(
                f"  {rating}: {count}/{len(values)} ({safediv(count, len(values)):.2%})"
            )
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
            rating=rating_to_binary(demo.rating),
        )
        for demo in demonstrations
    )
    return f"\n{"-" * 50}\n".join(output_all)


class GPTFull(BaseModel):
    """Decision on if the paper should be published and the reason for the decision."""

    model_config = ConfigDict(frozen=True)

    rationale: Annotated[str, Field(description="How you reached your novelty rating.")]
    rating: Annotated[
        int,
        Field(
            description="The novelty rating - how novel the paper is judged to be. Must"
            " be between 1 and 5.",
        ),
    ]

    @classmethod
    def error(cls) -> Self:
        """Output value for when there's an error."""
        return cls(rationale="<error>", rating=1)


def _load_demonstrations() -> dict[str, list[Demonstration]]:
    """Load demonstration files from the gpt.prompts package."""
    names = ["eval_demonstrations_4.json", "eval_demonstrations_10.json"]
    files = [
        file
        for file in resources.files("paper.gpt.prompts").iterdir()
        if file.name in names
    ]
    return {
        cast(Path, file).stem: load_data(file.read_bytes(), Demonstration)
        for file in files
    }


EVALUATE_DEMONSTRATIONS = _load_demonstrations()


def fix_classified_rating(classified: GPTFull) -> GPTFull:
    """Fix classified rating if out of range by clamping to [1, 5].

    Args:
        classified: Classified result to be checked.

    Returns:
        Same input if valid rating, or new object with fixed rating.
    """
    if classified.rating in range(1, 6):
        return classified

    logger.warning("Invalid rating: %d. Clamping to 1-5.", classified.rating)
    clamped_rating = max(1, min(classified.rating, 5))
    return replace_fields(classified, rating=clamped_rating)


def rating_to_binary(rating: int, mode: int | None = None) -> int:
    """Apply mode conversion to rating.

    Args:
        rating: Rating to be converted.
        mode:
            - 0: keep original integer rating.
            - 1-5: keep `rating >= x` as "positive", the rest as "negative"
            - -1: ternary: 1-2 is 1 (negative), 3 is 2 (neutral), 4-5 is 3 (positive).

            If `mode` is None, the value will be taken from the `MODE` environment
            variable.

    Returns:
        Converted rating, given mode.
    """
    if mode is None:
        mode = int(os.getenv("MODE", "0"))
    # Original integer mode.
    if mode == 0:
        return rating

    # Trinary mode.
    if mode == -1:
        if rating in [1, 2]:
            return 1
        if rating == 3:
            return 2
        return 3

    # Binary mode.
    return int(rating >= mode)
