"""Common types used to represent entities in the ASAP dataset processing."""

from __future__ import annotations

import enum
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from paper.util import hashstr
from paper.util.serde import Record


# Models from the ASAP files after exctraction (e.g. asap_filtered.json)
class ContextPolarity(enum.StrEnum):
    POSITIVE = enum.auto()
    NEGATIVE = enum.auto()
    NEUTRAL = enum.auto()


class ContextPolarityBinary(enum.StrEnum):
    POSITIVE = enum.auto()
    NEGATIVE = enum.auto()

    @classmethod
    def from_trinary(cls, polarity: ContextPolarity) -> ContextPolarityBinary:
        return (
            cls.POSITIVE
            if polarity in (ContextPolarity.POSITIVE, ContextPolarity.NEUTRAL)
            else cls.NEGATIVE
        )


class CitationContext(BaseModel):
    sentence: str = Field(description="Context sentence the from ASAP data")
    polarity: ContextPolarity | None = Field(
        description="Polarity of the citation context between main and reference papers."
    )


class PaperReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the citation in the paper references")
    year: int = Field(description="Year of publication")
    authors: Sequence[str] = Field(description="Author names")
    contexts: Sequence[CitationContext] = Field(
        description="Citation context with optional polarity evaluation"
    )


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str = Field(description="Section heading")
    text: str = Field(description="Section full text")


class PaperReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    rating: int = Field(description="Rating given by the review (1 to 10)")
    rationale: str = Field(description="Explanation given for the rating")


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[PaperReview] = Field(description="Feedback from a reviewer")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[PaperReference] = Field(
        description="References made in the paper"
    )


# Models after enrichment of references with data from the S2 API
class ReferenceWithAbstract(PaperReference):
    """ASAP reference with the added abstract and the original S2 title.

    `s2title` is the title in the S2 data for the best match. It can be used to match
    back to the original S2 file if desired.
    """

    abstract: str = Field(description="Abstract text")
    s2title: str = Field(description="Title from the S2 data")


class PaperWithFullReference(Record):
    """Paper from ASAP where the references contain their abstract."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[PaperReview] = Field(description="Feedback from a reviewer")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceWithAbstract] = Field(
        description="References made in the paper with their abstracts"
    )

    @property
    def id(self) -> str:
        return hashstr(self.title + self.abstract)


class TLDR(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    text: str | None


class S2Paper(BaseModel):
    """Paper from the S2 API."""

    model_config = ConfigDict(frozen=True)

    title_query: str = Field(description="Title used in the API query (from ASAP)")
    title: str = Field(description="Title from the S2 data")
    paper_id: str = Field(
        alias="paperId",
        description="Semantic Scholar's primary unique identifier for a paper",
    )
    corpus_id: int | None = Field(
        alias="corpusId",
        description="Semantic Scholar's secondary unique identifier for a paper",
    )
    url: str | None = Field(
        description="URL of the paper on the Semantic Scholar website"
    )
    abstract: str = Field(description="Abstract text")
    year: int | None = Field(description="Year the paper was published")
    reference_count: int = Field(
        alias="referenceCount", description="Number of papers this paper references"
    )
    citation_count: int = Field(
        alias="citationCount", description="Number of other papers that cite this paper"
    )
    influential_citation_count: int = Field(
        alias="influentialCitationCount",
        description="Number of influential papers (see docstring) that cite this paper",
    )
    tldr: TLDR | None = Field(description="Machine-generated summary of this paper")
    authors: Sequence[S2Author] | None = Field(description="Paper authors")


class S2Author(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str | None = Field(description="Author's name")
    author_id: str | None = Field(
        alias="authorId", description="Semantic Scholar's unique ID for the author"
    )


class ReferenceEnriched(PaperReference):
    """ASAP reference with the added data from the S2 API and the original S2 title.

    Attributes:
        abstract: Full text of the paper abstract.
        s2title: the title for the reference in the S2 API. Could differ from the
            reference title, so I keep this here in case we need to match back to the
            S2 data again.
        reference_count: How many references the current paper cites (outgoing)
        citation_count: How many papers cite the current paper (incoming)
        influential_citation_count: See https://www.semanticscholar.org/faq#influential-citations
        tldr: Machine-generated TLDR of the paper. Not available for everything.
    """

    abstract: str
    s2title: str
    reference_count: int
    citation_count: int
    influential_citation_count: int
    tldr: TLDR | None


class PaperWithReferenceEnriched(BaseModel):
    """Paper from ASAP where the references contain extra data from the S2 API."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[PaperReview] = Field(description="Feedback from a reviewer")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceEnriched] = Field(
        description="References made in the paper with their abstracts"
    )
