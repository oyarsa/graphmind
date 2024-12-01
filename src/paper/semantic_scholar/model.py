"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

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


class ASAPPaperMaybeS2(asap.Paper):
    """ASAP paper that may or may not have associated S2 paper information."""

    s2: Paper | None
    fuzz_ratio: int

    @classmethod
    def from_asap(cls, asap: asap.Paper, s2_result: dict[str, Any] | None) -> Self:
        """Create new paper from an existing ASAP paper and the S2 API result."""
        return cls(
            title=asap.title,
            abstract=asap.abstract,
            reviews=asap.reviews,
            sections=asap.sections,
            approval=asap.approval,
            references=asap.references,
            s2=Paper.model_validate(s2_result) if s2_result else None,
            fuzz_ratio=title_ratio(asap.title, s2_result["title"]) if s2_result else 0,
        )


class ASAPPaperWithS2(asap.Paper):
    """ASAP paper with associated S2 paper information."""

    s2: Paper
    fuzz_ratio: int

    @classmethod
    def from_maybe(cls, maybe: ASAPPaperMaybeS2, s2: Paper) -> Self:
        """Create an object that _definitely_ has S2 data from one that _maybe_ does."""
        return cls(
            title=maybe.title,
            abstract=maybe.abstract,
            reviews=maybe.reviews,
            sections=maybe.sections,
            approval=maybe.approval,
            references=maybe.references,
            s2=s2,
            fuzz_ratio=maybe.fuzz_ratio,
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
    sections: Sequence[asap.PaperSection] = Field(
        description="Sections in the paper text"
    )
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[asap.S2Paper] = Field(
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
