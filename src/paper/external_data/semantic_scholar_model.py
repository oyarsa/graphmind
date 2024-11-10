"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    """Paper returned by the Semantic Scholar API. Everything's optional but `corpusId`.

    This is to avoid validation errors in the middle of the download. We'll only save
    those with non-empty `abstract`, though.
    """

    model_config = ConfigDict(frozen=True)

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
    model_config = ConfigDict(frozen=True)

    # Author's name.
    name: str | None
    # Semantic Scholar's unique ID for the author.
    author_id: Annotated[str | None, Field(alias="authorId")]
