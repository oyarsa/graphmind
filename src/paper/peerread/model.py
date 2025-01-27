"""Types used to represent entities in the PeerRead dataset."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from functools import cached_property
from typing import Annotated, override

from pydantic import BaseModel, ConfigDict, Field, computed_field

from paper.util import hashstr
from paper.util.serde import Record


class PaperSection(BaseModel):
    """Section of a PeerRead full paper with its heading and context text."""

    model_config = ConfigDict(frozen=True)

    heading: Annotated[str, Field(description="Section heading")]
    text: Annotated[str, Field(description="Section full text")]


class PaperReview(BaseModel):
    """Peer review of a PeerRead paper with a novelty rating and rationale."""

    model_config = ConfigDict(frozen=True)

    rating: Annotated[
        int,
        Field(description="Novelty rating given by the reviewer (1 to 5)"),
    ]
    confidence: Annotated[int | None, Field(description="Confidence from the reviewer")]
    rationale: Annotated[str, Field(description="Explanation given for the rating")]


class ContextPolarity(StrEnum):
    """Binary polarity where "neutral" is converted to "positive"."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class CitationContext(BaseModel):
    """Citation context sentence with its (optional) predicted/annotated polarity."""

    sentence: Annotated[
        str, Field(description="Context sentence the from PeerRead data")
    ]
    polarity: Annotated[
        ContextPolarity | None,
        Field(
            description="Polarity of the citation context between main and reference papers."
        ),
    ]


class PaperReference(BaseModel):
    """Paper metadata with its contexts."""

    model_config = ConfigDict(frozen=True)

    title: Annotated[
        str, Field(description="Title of the citation in the paper references")
    ]
    year: Annotated[int, Field(description="Year of publication")]
    authors: Annotated[Sequence[str], Field(description="Author names")]
    contexts: Annotated[
        Sequence[CitationContext],
        Field(description="Citation context with optional polarity evaluation"),
    ]


class Paper(Record):
    """PeerRead paper with all available fields."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    references: Annotated[
        Sequence[PaperReference], Field(description="References made in the paper")
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    @computed_field
    @cached_property
    def review(self) -> PaperReview:
        """Get the review with median rating, breaking ties by confidence and rationale.

        Returns:
            The review with the median rating after applying tiebreakers.
        """
        reviews = self.reviews
        if not reviews:
            raise ValueError("Cannot get median from empty list")

        # Sort all reviews by rating
        sorted_reviews = sorted(reviews, key=lambda x: x.rating)
        median_idx = len(sorted_reviews) // 2

        # Get all reviews with the median rating
        median_rating = sorted_reviews[median_idx].rating
        median_reviews = [r for r in reviews if r.rating == median_rating]

        if len(median_reviews) == 1:
            return median_reviews[0]

        # Break ties by confidence (if present)
        if reviews_with_confidence := [
            r for r in median_reviews if r.confidence is not None
        ]:
            # 0 won't be used because we know all reviews have confidence
            return max(reviews_with_confidence, key=lambda x: x.confidence or 0)

        # If no confidence values or all tied, sort by rationale
        return min(median_reviews, key=lambda x: x.rationale)

    @computed_field
    @property
    def rating(self) -> int:
        """Rating from main review (1 to 5)."""
        return self.review.rating

    @computed_field
    @property
    def rationale(self) -> str:
        """Rationale from main review."""
        return self.review.rationale

    @property
    def main_text(self) -> str:
        """Join all paper sections to form the main text."""
        return "\n".join(s.text for s in self.sections)

    def __str__(self) -> str:
        """Display title, abstract, rating scores and count of words in main text."""
        main_text_words_num = len(self.main_text.split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {[r.rating for r in self.reviews]}\n"
        )
