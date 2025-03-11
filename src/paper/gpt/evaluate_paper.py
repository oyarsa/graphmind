"""Tools for evaluating paper novelty, displaying and calculating metrics."""

import logging
import statistics
from collections.abc import Sequence
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Annotated, Self, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

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
                "y_true": paper.label,
                "rationale_true": paper.rationale,
                "y_pred": y_pred,
                "rationale_pred": rationale_pred,
            }
        )


class RatingStats(BaseModel):
    """Mean/stdev/median stats on novelty ratings."""

    model_config = ConfigDict(frozen=True)

    mean: float
    stdev: float
    median: float

    @classmethod
    def calc(cls, values: Sequence[int]) -> Self:
        """Calculate stats from sequence of values."""
        return cls(
            mean=statistics.mean(values),
            stdev=statistics.stdev(values) if len(values) > 2 else 0,
            median=statistics.median(values),
        )

    def __str__(self) -> str:
        """Format stats one per line."""
        return "\n".join([
            f"mean   : {self.mean:.4f}",
            f"stdev  : {self.stdev:.4f}",
            f"median : {self.median:.4f}",
        ])


class PaperMetrics(evaluation_metrics.Metrics):
    """Evaluation metrics with total API cost."""

    cost: float
    stats_pred: RatingStats
    stats_true: RatingStats

    @classmethod
    def from_eval(
        cls,
        eval: evaluation_metrics.Metrics,
        cost: float,
        y_true: Sequence[int],
        y_pred: Sequence[int],
    ) -> Self:
        """Build metrics with cost from standard evaluation metrics."""
        return cls.model_validate(
            eval.model_dump()
            | {
                "cost": cost,
                "stats_pred": RatingStats.calc(y_pred),
                "stats_true": RatingStats.calc(y_true),
            }
        )


def calculate_paper_metrics(
    papers: Sequence[PaperResult], cost: float
) -> evaluation_metrics.Metrics:
    """Calculate evaluation metrics, including how much it cost.

    See also `paper.evaluation_metrics.calculate_metrics`.
    """
    y_pred = [p.y_pred for p in papers]
    y_true = [p.y_true for p in papers]

    return PaperMetrics.from_eval(
        evaluation_metrics.calculate_metrics(y_true, y_pred), cost, y_true, y_pred
    )


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
    for values, section in [(y_true, "Gold"), (y_pred, "Predicted")]:
        output.append(f"\n{section} distribution:")
        for label in metrics.mode.labels():
            count = sum(y == label for y in values)
            output.append(
                f"  {label}: {count}/{len(values)} ({safediv(count, len(values)):.2%})"
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

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)


def get_demonstrations(demonstrations_key: str | None, prompt_key: str) -> str:
    """Get demonstrations rendered as a string.

    Args:
        demonstrations_key: Key of the demonstrations in `EVALUATE_DEMONSTRATIONS`.
            If None, this will return an empty string.
        prompt_key: Key of the prompt to use to render the demonstrations.

    Returns:
        Demonstrations rendered as a string if `demonstrations_key` is not None. Else,
        return an empty string.
    """
    if not demonstrations_key:
        return ""

    return format_demonstrations(
        demonstrations=EVALUATE_DEMONSTRATIONS[demonstrations_key],
        prompt=EVALUATE_DEMONSTRATION_PROMPTS[prompt_key],
    )


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
            rating=demo.rating,
        )
        for demo in demonstrations
    )
    return f"\n{'-' * 50}\n".join(output_all)


class GPTFull(BaseModel):
    """Decision on if the paper should be published and the reason for the decision."""

    model_config = ConfigDict(frozen=True)

    rationale: Annotated[str, Field(description="How you reached your novelty rating.")]
    label: Annotated[
        int,
        Field(description="1 if the paper is novel, or 0 if it's not novel."),
    ]

    @classmethod
    def error(cls) -> Self:
        """Output value for when there's an error."""
        return cls(rationale="<error>", label=0)

    def is_valid(self) -> bool:
        """Check if instance is valid."""
        return self.rationale != "<error>"


def _load_demonstrations() -> dict[str, list[Demonstration]]:
    """Load demonstration files from the gpt.demonstrations package."""
    return {
        cast(Path, file).stem: load_data(file.read_bytes(), Demonstration)
        for file in resources.files("paper.gpt.demonstrations").iterdir()
        if file.name.endswith(".json")
    }


EVALUATE_DEMONSTRATIONS = _load_demonstrations()
"""Available demonstrations from `paper.gpt.demonstrations`."""


def fix_evaluated_rating(evaluated: GPTFull) -> GPTFull:
    """Fix evaluated label if out of range by converting to 0/1.

    Any label that isn't 1 will be treated as 0.

    Args:
        evaluated: Evaluation result to be checked.

    Returns:
        Same input if valid label, or new object with fixed label.
    """
    if evaluated.label not in [0, 1]:
        logger.warning("Invalid label: %d. Converting to 0/1", evaluated.label)

    return replace_fields(evaluated, label=evaluated.label == 1)


class RatingMode(StrEnum):
    """What rating mode to use.

    Original: keeps the original 1-5 rating.
    Binary: converts 1-3 to 0 and 4-5 to 1.
    Ternary: convert 1-2 to 1 (negative), 3 to 2 (neutral) and 4-5 to 3 (positive).
    """

    ORIGINAL = "original"
    BINARY = "binary"
    TRINARY = "trinary"


def apply_rating_mode(rating: int, mode: RatingMode) -> int:
    """Apply mode conversion to rating.

    Args:
        rating: Rating to be converted.
        mode: Rating mode to apply.

    Returns:
        Converted rating, given mode.
    """
    match mode:
        # Original integer mode.
        case RatingMode.ORIGINAL:
            return rating
        case RatingMode.TRINARY:
            if rating in [1, 2]:
                return 1
            if rating == 3:
                return 2
            return 3
        case RatingMode.BINARY:
            return int(rating >= 4)
