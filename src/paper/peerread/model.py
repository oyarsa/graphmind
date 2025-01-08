"""Types used to represent entities in the PeerRead dataset."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, override

from pydantic import BaseModel, ConfigDict, Field

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
        int, Field(description="Novelty rating given by the reviewer (1 to 5)")
    ]
    confidence: Annotated[int | None, Field(description="Confidence from the reviwer")]
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
