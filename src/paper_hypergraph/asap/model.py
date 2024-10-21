from __future__ import annotations

import enum
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


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

    def __bool__(self) -> bool:
        return self is self.POSITIVE


class ContextAnnotated(BaseModel):
    regular: str = Field(description="Regular context the from ASAP data")
    expanded: str = Field(description="Expanded context")
    polarity: ContextPolarity | None = None


class PaperReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the citation in the paper references")
    year: int = Field(description="Year of publication")
    authors: Sequence[str] = Field(description="Author names")
    contexts: Sequence[str] = Field(description="Citation contexts from this reference")
    contexts_expanded: Sequence[str] = Field(
        description="Citation contexts from this reference (expanded in paragraph)"
    )
    # TODO: Rework the types here. This feels like a hack because ASAP has nothing
    # to do with the annotation.
    contexts_annotated: Sequence[ContextAnnotated] | None = Field(
        default=None, description="Citation context with golden polarity evaluation"
    )


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str = Field(description="Section heading")
    text: str = Field(description="Section full text")


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
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


class PaperWithFullReference(BaseModel):
    """Paper from ASAP where the references contain their abstract."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceWithAbstract] = Field(
        description="References made in the paper with their abstracts"
    )


class TLDR(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    text: str | None


class S2Paper(BaseModel):
    """Paper from the S2 API.

    Attributes:
        title_query: The original title used to query the API.
        title: Actual title of the paper in the API.
        abstract: Full text of the paper abstract.
        reference_count: How many references the current paper cites (outgoing).
        citation_count: How many papers cite the current paper (incoming).
        influential_citation_count: See https://www.semanticscholar.org/faq#influential-citations.
        tldr: Machine-generated TLDR of the paper. Not available for everything.

    NB: We got more data from the API, but this is what's relevant here. See also
    `paper_hypergraph.external_data.semantic_scholar`.
    """

    model_config = ConfigDict(frozen=True)

    title_query: str = Field(description="Title used in the API query (from ASAP)")
    title: str = Field(description="Title from the S2 data")
    abstract: str = Field(description="Abstract text")
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

    title: str
    abstract: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]
    approval: bool
    references: Sequence[ReferenceEnriched]
