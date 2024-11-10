"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field

from paper.asap.model import Paper as ASAPPaper
from paper.util import fuzzy_partial_ratio


class Paper(BaseModel):
    """Paper returned by the Semantic Scholar API. Everything's optional but `paperId`.

    This is to avoid validation errors in the middle of the download. We'll only save
    those with non-empty `abstract`, though.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

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
    tldr: Tldr | None
    # Paper authors.
    authors: Sequence[Author] | None


class Tldr(BaseModel):
    model_config = ConfigDict(frozen=True)

    # The tldr model version number: https://github.com/allenai/scitldr
    model: str | None
    # The tldr paper summary.
    text: str | None


class Author(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Author's name.
    name: str | None
    # Semantic Scholar's unique ID for the author.
    author_id: Annotated[str | None, Field(alias="authorId")]


class ASAPPaperMaybeS2(ASAPPaper):
    s2: Paper | None
    fuzz_ratio: int

    @classmethod
    def from_asap(cls, asap: ASAPPaper, s2_result: dict[str, Any] | None) -> Self:
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


class ASAPPaperWithS2(ASAPPaper):
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


def title_ratio(title1: str, title2: str) -> int:
    """Calculate fuzzy ratio between paper titles.

    Calculates the partial ratio between clean titles. Clean titles are case-folded and
    stripped.
    """
    return fuzzy_partial_ratio(_clean_title(title1), _clean_title(title2))


def _clean_title(title: str) -> str:
    return title.casefold().strip()
