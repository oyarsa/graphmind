"""Common types used to represent entities in the ASAP dataset processing."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from paper.util import hashstr
from paper.util.serde import Record


# Models from the ASAP files after exctraction (e.g. asap_filtered.json)
class ContextPolarityTrinary(StrEnum):
    """Human-classified polarity of citation context."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ContextPolarity(StrEnum):
    """Binary polarity where "neutral" is converted to "positive"."""

    POSITIVE = "positive"
    NEGATIVE = "negative"

    @classmethod
    def from_trinary(cls, polarity: ContextPolarityTrinary) -> ContextPolarity:
        """Converts "neutral" polarity to "positive"."""
        return (
            cls.POSITIVE
            if polarity
            in (ContextPolarityTrinary.POSITIVE, ContextPolarityTrinary.NEUTRAL)
            else cls.NEGATIVE
        )


class CitationContext(BaseModel):
    """Citation context sentence with its (optional) predicted/annotated polarity."""

    sentence: str = Field(description="Context sentence the from ASAP data")
    polarity: ContextPolarityTrinary | None = Field(
        description="Polarity of the citation context between main and reference papers."
    )


class PaperReference(BaseModel):
    """Paper metadata with its contexts."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the citation in the paper references")
    year: int = Field(description="Year of publication")
    authors: Sequence[str] = Field(description="Author names")
    contexts: Sequence[CitationContext] = Field(
        description="Citation context with optional polarity evaluation"
    )


class PaperSection(BaseModel):
    """Section of an ASAP full paper with its heading and context text."""

    model_config = ConfigDict(frozen=True)

    heading: str = Field(description="Section heading")
    text: str = Field(description="Section full text")


class PaperReview(BaseModel):
    """Peer review of an ASAP paper with a 1-10 rating and rationale."""

    model_config = ConfigDict(frozen=True)

    rating: int = Field(description="Rating given by the review (1 to 10)")
    rationale: str = Field(description="Explanation given for the rating")


class Paper(Record):
    """ASAP paper with all available fields."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[PaperReview] = Field(description="Feedback from a reviewer")
    authors: Sequence[str] = Field(description="Names of the authors")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[PaperReference] = Field(
        description="References made in the paper"
    )

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)
