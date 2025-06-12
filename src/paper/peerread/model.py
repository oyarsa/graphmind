"""Types used to represent entities in the PeerRead dataset."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Annotated, Self, override

from pydantic import BaseModel, Field, computed_field, model_validator

from paper.types import Immutable
from paper.util import fix_spaces_before_punctuation, hashstr
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper.semantic_scholar.model import PaperFromPeerRead


class PaperSection(Immutable):
    """Section of a PeerRead full paper with its heading and context text."""

    heading: Annotated[str, Field(description="Section heading")]
    text: Annotated[str, Field(description="Section full text")]


class PaperReview(Immutable):
    """Peer review of a PeerRead paper with a novelty rating and rationale."""

    rating: Annotated[
        int,
        Field(description="Novelty rating given by the reviewer (1 to 5)"),
    ]
    confidence: Annotated[int | None, Field(description="Confidence from the reviewer")]
    rationale: Annotated[str, Field(description="Explanation given for the rating")]
    other_ratings: Annotated[
        dict[str, int], Field(description="Other available ratings")
    ] = {}  # noqa: RUF012

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)


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

    @model_validator(mode="after")
    def truncate_sentence(self) -> Self:
        """Truncate the context sentence to 256 characters."""
        self.sentence = self.sentence[:256]
        return self


class PaperReference(Immutable):
    """Paper metadata with its contexts."""

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
    ] = []
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
    year: Annotated[int | None, Field(description="Paper publication year")] = None

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    @computed_field
    @cached_property
    def review(self) -> PaperReview | None:
        """Get the review with median rating, breaking ties by confidence and rationale.

        Returns:
            The review with the median rating after applying tiebreakers, or None if no
            reviews.
        """
        reviews = self.reviews
        if not reviews:
            return None

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
        """Rating from main review (1 to 5), or 0 if no reviews."""
        if self.review is None:
            return 0
        return self.review.rating

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @computed_field
    @property
    def rationale(self) -> str:
        """Rationale from main review, or empty string if no reviews."""
        if self.review is None:
            return ""
        return self.review.rationale

    def main_text(self) -> str:
        """Join all paper sections to form the main text."""
        return clean_maintext("\n".join(s.text for s in self.sections))

    @classmethod
    def from_s2(
        cls,
        s2_paper: PaperFromPeerRead,
        *,
        sections: Sequence[PaperSection],
        references: Sequence[PaperReference],
        conference: str,
    ) -> Self:
        """Create Paper from Semantic Scholar data without reviews.

        Args:
            s2_paper: Paper data from Semantic Scholar API.
            sections: Paper sections (from arXiv LaTeX parsing).
            references: Paper references (from arXiv LaTeX parsing).
            conference: Conference name (can be inferred from venue).

        Returns:
            Paper instance with S2 metadata and empty reviews.
        """
        author_names = [author.name for author in s2_paper.authors or [] if author.name]

        return cls(
            title=s2_paper.title,
            abstract=s2_paper.abstract,
            reviews=[],  # No reviews from S2
            authors=author_names,
            sections=sections,
            approval=None,  # No approval data from S2
            references=references,
            conference=conference,
            year=s2_paper.year,
        )

    def __str__(self) -> str:
        """Display title, abstract, rating scores and count of words in main text."""
        main_text_words_num = len(self.main_text().split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {[r.rating for r in self.reviews]}\n"
        )


def remove_line_numbers(content: str) -> str:
    """Remove line numbers at the start of lines."""
    lines = (re.sub(r"^\d+\s*", "", line) for line in content.splitlines())
    return "\n".join(lines)


def compress_whitespace(content: str) -> str:
    """Remove multiple consecutive spaces and normalize newlines."""
    content = re.sub(r"\n{3,}", "\n\n", content)
    return re.sub(r" {2,}", " ", content)


def remove_page_numbers(content: str) -> str:
    """Remove isolated page numbers."""
    return re.sub(r"^\s*\d+\s*$", "", content, flags=re.MULTILINE)


def normalise_paragraphs(content: str) -> str:
    """Normalize paragraph breaks and whitespace."""
    paragraphs = (p.strip() for p in content.split("\n\n"))
    return "\n\n".join(paragraphs)


def clean_maintext(content: str) -> str:
    """Clean up PDF content using a series of transformations."""
    cleanup = [
        remove_line_numbers,
        remove_page_numbers,
        compress_whitespace,
        fix_spaces_before_punctuation,
        normalise_paragraphs,
    ]
    return reduce(lambda content, fn: fn(content), cleanup, content)
