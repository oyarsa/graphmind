"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self, override

from pydantic import BaseModel, ConfigDict, Field

from paper import asap
from paper.util import fuzzy_partial_ratio, hashstr
from paper.util.serde import Record


class Paper(Record):
    """Paper returned by the Semantic Scholar API. Everything's optional but `paperId`.

    This is to avoid validation errors in the middle of the download. We'll only save
    those with non-empty `abstract`, though.
    """

    # Semantic Scholar's primary unique identifier for a paper.
    paper_id: Annotated[str, Field(alias="paperId")]
    # Semantic Scholar's secondary unique identifier for a paper.
    corpus_id: Annotated[int | None, Field(alias="corpusId")]
    # URL of the paper on the Semantic Scholar website.
    url: str | None
    # Title of the paper.
    title: str | None
    # The paper's abstract. Note that due to legal reasons, this may be missing even if
    # we display an abstract on the website.
    abstract: str | None
    # The year the paper was published.
    year: int | None
    # The total number of papers this paper references.
    reference_count: Annotated[int | None, Field(alias="referenceCount")]
    # The total number of papers that reference this paper.
    citation_count: Annotated[int | None, Field(alias="citationCount")]
    # A subset of the citation count, where the cited publication has a significant
    # impact on the citing publication.
    influential_citation_count: Annotated[
        int | None, Field(alias="influentialCitationCount")
    ]
    # The tldr paper summary.
    tldr: Tldr | None = None
    # Paper authors.
    authors: Sequence[Author] | None

    @property
    def id(self) -> str:
        """Identify the S2 paper by the S2 paper ID."""
        return self.paper_id


class Tldr(BaseModel):
    """AI-generated one-sentence paper summary."""

    model_config = ConfigDict(frozen=True)

    # The tldr model version number: https://github.com/allenai/scitldr
    model: str | None
    # The tldr paper summary.
    text: str | None


class Author(BaseModel):
    """Author information from the S2 API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Author's name.
    name: str | None
    # Semantic Scholar's unique ID for the author.
    author_id: Annotated[str | None, Field(alias="authorId")]


class PaperFromASAP(Record):
    """Paper from querying an ASAP paper the S2 API with useful fields required.

    The original Paper type has most things optional because we don't know what the API
    will return for each individual paper. This type, on the other hand, requires all
    the useful fields.
    """

    title_asap: str = Field(
        description="Title used in the API query (from ASAP)", alias="title_query"
    )
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
    tldr: Tldr | None = Field(description="Machine-generated summary of this paper")
    authors: Sequence[Author] | None = Field(description="Paper authors")

    @property
    def id(self) -> str:
        """Identify paper by the S2 API paper ID."""
        return self.paper_id


class S2Reference(PaperFromASAP):
    """S2 paper as a reference with the original contexts."""

    contexts: Sequence[asap.CitationContext]

    @classmethod
    def from_(
        cls, paper: PaperFromASAP, *, contexts: Sequence[asap.CitationContext]
    ) -> Self:
        """Create new instance by copying data from S2Paper, in addition to the contexts."""
        return cls.model_validate(paper.model_dump() | {"contexts": contexts})


class PaperWithS2Refs(Record):
    """ASAP main paper where references have the full S2 data as well as their contexts."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[asap.PaperReview] = Field(description="Feedback from a reviewer")
    authors: Sequence[str] = Field(description="Names of the authors")
    sections: Sequence[asap.PaperSection] = Field(
        description="Sections in the paper text"
    )
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[S2Reference] = Field(
        description="References from the paper with full S2 data and citation contexts."
    )

    @property
    def id(self) -> str:
        """Identify an ASAP by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)


class ASAPPaperWithS2(asap.Paper):
    """ASAP paper with associated S2 paper information."""

    s2: PaperFromASAP
    fuzz_ratio: int

    @classmethod
    def from_asap(cls, asap: asap.Paper, s2_result: PaperFromASAP) -> Self:
        """Create new paper from an existing ASAP paper and the S2 API result."""
        return cls(
            title=asap.title,
            abstract=asap.abstract,
            reviews=asap.reviews,
            authors=asap.authors,
            sections=asap.sections,
            approval=asap.approval,
            references=asap.references,
            s2=s2_result,
            fuzz_ratio=title_ratio(asap.title, s2_result.title),
        )


class PaperWithRecommendations(BaseModel):
    """ASAP paper with a list of recommendations from the S2 API."""

    model_config = ConfigDict(frozen=True)

    main_paper: ASAPPaperWithS2
    recommendations: Sequence[Paper]


class PaperRecommended(Paper):
    """S2 paper recommended from ASAP papers.

    Attributes:
        sources_asap: ASAP titles for the the papers that led to this.
        sources_s2: S2 titles for the the papers that led to this.
    """

    sources_asap: Sequence[str]
    sources_s2: Sequence[str]


def title_ratio(title1: str, title2: str) -> int:
    """Calculate fuzzy ratio between paper titles.

    Calculates the partial ratio between clean titles. Clean titles are case-folded and
    stripped.
    """
    return fuzzy_partial_ratio(clean_title(title1), clean_title(title2))


def clean_title(title: str) -> str:
    """Clean a paper title by casefolding and removing extraneous whitespace."""
    return " ".join(title.casefold().strip().split())


class ASAPWithFullS2(Record):
    """ASAP main paper where references have the full S2 data."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[asap.PaperReview] = Field(description="Feedback from a reviewer")
    authors: Sequence[str] = Field(description="Names of the authors")
    sections: Sequence[asap.PaperSection] = Field(
        description="Sections in the paper text"
    )
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[PaperFromASAP] = Field(
        description="References made in the paper with full S2 data"
    )

    @property
    def id(self) -> str:
        """Identify an ASAP by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)


class PaperArea(Paper):
    """S2 paper with the areas that led to it."""

    model_config = ConfigDict(frozen=True)

    areas: Sequence[str]


class ReferenceEnriched(asap.PaperReference):
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
    tldr: Tldr | None


class PaperWithReferenceEnriched(BaseModel):
    """Paper from ASAP where the references contain extra data from the S2 API."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[asap.PaperReview] = Field(description="Feedback from a reviewer")
    authors: Sequence[str] = Field(description="Names of the authors")
    sections: Sequence[asap.PaperSection] = Field(
        description="Sections in the paper text"
    )
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceEnriched] = Field(
        description="References made in the paper with their abstracts"
    )


class ReferenceWithAbstract(asap.PaperReference):
    """ASAP reference with the added abstract and the original S2 title.

    `s2title` is the title in the S2 data for the best match. It can be used to match
    back to the original S2 file if desired.
    """

    abstract: str = Field(description="Abstract text")
    s2title: str = Field(description="Title from the S2 data")
    paper_id: str = Field(description="Paper ID in the S2 API")

    @property
    def id(self) -> str:
        """Identify the reference by its S2 API ID."""
        return self.paper_id


class PaperWithFullReference(Record):
    """Paper from ASAP where the references contain their abstract."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[asap.PaperReview] = Field(description="Feedback from a reviewer")
    authors: Sequence[str] = Field(description="Names of the authors")
    sections: Sequence[asap.PaperSection] = Field(
        description="Sections in the paper text"
    )
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceWithAbstract] = Field(
        description="References made in the paper with their abstracts"
    )

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)
