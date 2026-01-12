"""Tools for evaluating paper novelty, displaying and calculating metrics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Annotated, Protocol, Self, cast

from pydantic import Field, computed_field

from paper import semantic_scholar as s2
from paper.evaluation_metrics import RATING_LABELS
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.types import Immutable, PaperSource
from paper.util.serde import load_data

logger = logging.getLogger(__name__)


class PaperResult(s2.PaperWithS2Refs):
    """PeerRead paper with added novelty rating ground truth and GPT prediction."""

    y_true: Annotated[int, Field(description="Human annotation")]
    y_pred: Annotated[int, Field(description="Model prediction")]
    rationale_true: Annotated[str, Field(description="Human rationale annotation")]
    rationale_pred: Annotated[
        str, Field(description="Model rationale for the prediction")
    ]
    structured_evaluation: GPTStructured | None = None
    confidence: Annotated[
        float | None,
        Field(
            description="Confidence in the prediction from ensemble voting (0.0-1.0)."
        ),
    ] = None

    @classmethod
    def from_s2peer(
        cls,
        paper: s2.PaperWithS2Refs,
        y_pred: int,
        rationale_pred: str,
        structured_evaluation: GPTStructured | None = None,
        confidence: float | None = None,
    ) -> Self:
        """Construct `PaperResult` from the original paper and model predictions."""
        # Extract confidence from structured_evaluation if available
        if structured_evaluation is not None:
            confidence = structured_evaluation.confidence

        return cls.model_validate(
            paper.model_dump()
            | {
                "y_true": paper.rating,
                "rationale_true": paper.rationale,
                "y_pred": y_pred,
                "rationale_pred": rationale_pred,
                "structured_evaluation": structured_evaluation,
                "confidence": confidence,
            }
        )


EVALUATE_DEMONSTRATION_PROMPTS = load_prompts("eval_demonstrations")


class Demonstration(Immutable):
    """Paper for evaluation demos with full information and demonstration type."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Paper abstract")]
    text: Annotated[str, Field(description="Paper full main text")]
    rationale: Annotated[str, Field(description="Rationale given by a reviewer")]
    rating: Annotated[int, Field(description="Rating from the rationale")]

    @computed_field
    @property
    def label(self) -> int:
        """Return the rating as the label."""
        return self.rating


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
            label=demo.label,
        )
        for demo in demonstrations
    )
    return f"\n{'-' * 50}\n".join(output_all)


class GPTFull(Immutable):
    """Evaluation of paper novelty on a 1-5 scale."""

    label: Annotated[
        int,
        Field(
            description=(
                "Novelty rating from 1 to 5:\n"
                "1 = Not novel: Significant portions done before or done better\n"
                "2 = Minor improvement on familiar techniques\n"
                "3 = Notable extension of prior approaches\n"
                "4 = Substantially different from previous research\n"
                "5 = Significant new problem, technique, or insight"
            ),
            ge=1,
            le=5,
        ),
    ]
    rationale: Annotated[str, Field(description="How you reached your novelty rating.")]

    @classmethod
    def error(cls) -> Self:
        """Output value for when there's an error."""
        return cls(rationale="<error>", label=1)

    def is_valid(self) -> bool:
        """Check if instance is valid."""
        return self.rationale != "<error>"

    def with_confidence(self, confidence: float | None) -> GPTFullWithConfidence:
        """Create a GPTFullWithConfidence instance with the given confidence.

        Args:
            confidence: Confidence in the label from ensemble voting (0.0-1.0).

        Returns:
            A GPTFullWithConfidence instance with the confidence.
        """
        return GPTFullWithConfidence.from_(self, confidence=confidence)


class GPTFullWithConfidence(GPTFull):
    """GPTFull evaluation with confidence from ensemble voting."""

    confidence: Annotated[
        float | None,
        Field(description="Confidence in the label from ensemble voting (0.0-1.0)."),
    ] = None

    @classmethod
    def from_(cls, full: GPTFull, *, confidence: float | None = None) -> Self:
        """Create a GPTFullWithConfidence from GPTFull and confidence."""
        return cls.model_validate(full.model_dump() | {"confidence": confidence})


class EvidenceItem(Immutable):
    """Evidence item with paper citation information."""

    text: Annotated[str, Field(description="The evidence text or finding.")]
    paper_id: Annotated[
        str | None,
        Field(
            description="ID of the paper this evidence comes from (e.g., S2 paper ID)."
        ),
    ]
    paper_title: Annotated[
        str | None,
        Field(description="Title of the paper this evidence comes from."),
    ]
    source: Annotated[
        PaperSource | None,
        Field(
            description="Source of the related paper (citations or semantic similarity)."
        ),
    ]


class GPTStructuredRaw(Immutable):
    """Structured evaluation of paper novelty with detailed components. Raw version."""

    paper_summary: Annotated[
        str,
        Field(
            description="Brief summary of the paper's main contributions and approach."
        ),
    ]
    supporting_evidence: Annotated[
        Sequence[EvidenceItem],
        Field(
            description="List of evidence from related papers that support the paper's"
            " novelty."
        ),
    ]
    contradictory_evidence: Annotated[
        Sequence[EvidenceItem],
        Field(
            description="List of evidence from related papers that contradict the"
            " paper's novelty."
        ),
    ]
    key_comparisons: Annotated[
        Sequence[str],
        Field(
            description="Key technical comparisons that influenced the novelty decision."
        ),
    ]
    conclusion: Annotated[
        str,
        Field(
            description="Final assessment of the paper's novelty based on the evidence."
        ),
    ]
    label: Annotated[
        int,
        Field(
            description=(
                "Novelty rating from 1 to 5:\n"
                "1 = Not novel: Significant portions done before or done better\n"
                "2 = Minor improvement on familiar techniques\n"
                "3 = Notable extension of prior approaches\n"
                "4 = Substantially different from previous research\n"
                "5 = Significant new problem, technique, or insight"
            ),
            ge=1,
            le=5,
        ),
    ]

    @computed_field
    @property
    def rationale(self) -> str:
        """Derive textual rationale from structured components."""
        sections = [
            f"Paper Summary: {self.paper_summary}",
            "",
            "Supporting Evidence:",
        ]

        for evidence in self.supporting_evidence:
            evidence_text = f"- {evidence.text}"
            if evidence.paper_title:
                evidence_text += f" (from: {evidence.paper_title})"
            sections.append(evidence_text)

        sections.extend(["", "Contradictory Evidence:"])

        for evidence in self.contradictory_evidence:
            evidence_text = f"- {evidence.text}"
            if evidence.paper_title:
                evidence_text += f" (from: {evidence.paper_title})"
            sections.append(evidence_text)

        if self.key_comparisons:
            sections.extend(["", "Key Comparisons:"])
            sections.extend([f"- {comp}" for comp in self.key_comparisons])

        sections.extend(["", f"Conclusion: {self.conclusion}"])
        return "\n".join(sections)

    @classmethod
    def error(cls) -> Self:
        """Output value for when there's an error."""
        return cls(
            paper_summary="<error>",
            supporting_evidence=[],
            contradictory_evidence=[],
            key_comparisons=[],
            conclusion="<error>",
            label=1,
        )

    def is_valid(self) -> bool:
        """Check if instance is valid."""
        return self.paper_summary != "<error>" and self.conclusion != "<error>"

    def with_prob(self, probability: float | None) -> GPTStructured:
        """Create a `GPTStructured` instance with the given probability.

        Args:
            probability: Probability of the evaluation being correct, if available.

        Returns:
            A `GPTStructured` instance with the raw data and probability.
        """
        return GPTStructured.from_(self, probability=probability)

    def with_confidence(self, confidence: float | None) -> GPTStructured:
        """Create a GPTStructured instance with the given confidence.

        Args:
            confidence: Confidence in the label from ensemble voting (0.0-1.0).

        Returns:
            A GPTStructured instance with the confidence and None probability.
        """
        return GPTStructured.from_(self, confidence=confidence)


class GPTStructured(GPTStructuredRaw):
    """Structured evaluation of paper novelty with detailed components.

    Version with the evaluation probability and confidence.
    """

    probability: Annotated[
        float | None,
        Field(description="Probability of the evaluation being correct, if available."),
    ]
    confidence: Annotated[
        float | None,
        Field(description="Confidence in the label from ensemble voting (0.0-1.0)."),
    ] = None

    @classmethod
    def from_(
        cls,
        raw: GPTStructuredRaw,
        *,
        probability: float | None = None,
        confidence: float | None = None,
    ) -> Self:
        """Create a `GPTStructured` from a raw evaluation and probability/confidence.

        Args:
            raw: Raw structured evaluation data.
            probability: Probability of the evaluation being correct, if available.
            confidence: Confidence in the label from ensemble voting, if available.

        Returns:
            A `GPTStructured` instance with the raw data and probability/confidence.
        """
        return cls.model_validate(
            raw.model_dump() | {"probability": probability, "confidence": confidence}
        )


def _load_demonstrations() -> dict[str, list[Demonstration]]:
    """Load demonstration files from the gpt.demonstrations package."""
    return {
        cast(Path, file).stem: load_data(file.read_bytes(), Demonstration)
        for file in resources.files("paper.gpt.demonstrations").iterdir()
        if file.name.endswith(".json")
    }


EVALUATE_DEMONSTRATIONS = _load_demonstrations()
"""Available demonstrations from `paper.gpt.demonstrations`."""


class EvaluationResult(Protocol):
    """Result of evaluating a paper with novelty label and rationale."""

    @property
    def rationale(self) -> str:
        """Rationale for novelty label."""
        ...

    @property
    def label(self) -> int:
        """Novelty label."""
        ...


def fix_evaluated_rating(evaluated: EvaluationResult) -> GPTFull:
    """Fix evaluated label if out of range by clamping to valid range (1-5).

    Args:
        evaluated: Evaluation result to be checked.

    Returns:
        Same input if valid label, or new object with clamped label.
    """
    if evaluated.label in RATING_LABELS:
        return GPTFull(label=evaluated.label, rationale=evaluated.rationale)

    clamped_label = max(min(evaluated.label, max(RATING_LABELS)), min(RATING_LABELS))
    logger.warning(
        "Invalid label: %d. Clamping to %d (valid range: %s)",
        evaluated.label,
        clamped_label,
        RATING_LABELS,
    )
    return GPTFull(label=clamped_label, rationale=evaluated.rationale)


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
