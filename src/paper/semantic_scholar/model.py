"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from functools import cached_property
from typing import Annotated, Self, override

from pydantic import Field, computed_field, field_validator

from paper import peerread as pr
from paper.peerread.model import clean_maintext, scale_recommendation
from paper.types import Immutable
from paper.util import fuzzy_partial_ratio, hashstr
from paper.util.serde import Record


def _parse_publication_date(value: object) -> date | None:
    """Parse publication date values from S2 API payloads.

    Supports ISO date strings and datetime strings. If month precision is not present
    (for example year-only strings), returns None.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    value_date = value.split("T", maxsplit=1)[0]
    if len(value_date) != 10:
        return None

    try:
        return date.fromisoformat(value_date)
    except ValueError:
        return None


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
    # Full publication date from S2, when available.
    publication_date: Annotated[date | None, Field(alias="publicationDate")] = None
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
    # Publication venue name.
    venue: str | None = None

    @field_validator("publication_date", mode="before")
    @classmethod
    def parse_publication_date(cls, value: object) -> date | None:
        """Parse publication_date safely without failing model validation."""
        return _parse_publication_date(value)

    @property
    def id(self) -> str:
        """Identify the S2 paper by the S2 paper ID."""
        return self.paper_id


class Tldr(Immutable):
    """AI-generated one-sentence paper summary."""

    # The tldr model version number: https://github.com/allenai/scitldr
    model: str | None
    # The tldr paper summary.
    text: str | None


class Author(Immutable):
    """Author information from the S2 API."""

    # Author's name.
    name: str | None
    # Semantic Scholar's unique ID for the author.
    author_id: Annotated[str | None, Field(alias="authorId")]


class PaperFromPeerRead(Record):
    """Paper from querying an PeerRead paper the S2 API with useful fields required.

    The original Paper type has most things optional because we don't know what the API
    will return for each individual paper. This type, on the other hand, requires all
    the useful fields.
    """

    title_peer: Annotated[
        str,
        Field(
            description="Title used in the API query (from PeerRead)",
            alias="title_query",
        ),
    ]
    title: Annotated[str, Field(description="Title from the S2 data")]
    paper_id: Annotated[
        str,
        Field(
            alias="paperId",
            description="Semantic Scholar's primary unique identifier for a paper",
        ),
    ]
    corpus_id: Annotated[
        int | None,
        Field(
            alias="corpusId",
            description="Semantic Scholar's secondary unique identifier for a paper",
        ),
    ]
    url: Annotated[
        str | None,
        Field(description="URL of the paper on the Semantic Scholar website"),
    ]
    abstract: Annotated[str, Field(description="Abstract text")]
    year: Annotated[int | None, Field(description="Year the paper was published")]
    publication_date: Annotated[
        date | None,
        Field(
            alias="publicationDate",
            description="Publication date from S2 API, when available",
        ),
    ] = None
    reference_count: Annotated[
        int,
        Field(
            alias="referenceCount", description="Number of papers this paper references"
        ),
    ]
    citation_count: Annotated[
        int,
        Field(
            alias="citationCount",
            description="Number of other papers that cite this paper",
        ),
    ]
    influential_citation_count: Annotated[
        int,
        Field(
            alias="influentialCitationCount",
            description="Number of influential papers (see docstring) that cite this paper",
        ),
    ]
    tldr: Annotated[
        Tldr | None, Field(description="Machine-generated summary of this paper")
    ] = None
    authors: Annotated[Sequence[Author] | None, Field(description="Paper authors")]
    venue: Annotated[str | None, Field(description="Publication venue name")] = None

    @field_validator("abstract", mode="before")
    @classmethod
    def replace_none_abstract(cls, v: str | None) -> str:
        """Replace None abstract with empty string."""
        return v if v is not None else ""

    @field_validator("publication_date", mode="before")
    @classmethod
    def parse_publication_date(cls, value: object) -> date | None:
        """Parse publication_date safely without failing model validation."""
        return _parse_publication_date(value)

    @property
    def id(self) -> str:
        """Identify paper by the S2 API paper ID."""
        return self.paper_id


class S2Reference(PaperFromPeerRead):
    """S2 paper as a reference with the original contexts."""

    contexts: Sequence[pr.CitationContext]
    citation_key: str = ""

    @classmethod
    def from_(
        cls,
        paper: PaperFromPeerRead,
        *,
        contexts: Sequence[pr.CitationContext],
        citation_key: str = "",
    ) -> Self:
        """Create new instance by copying data from S2Paper, in addition to the contexts."""
        return cls.model_validate(
            paper.model_dump() | {"contexts": contexts, "citation_key": citation_key}
        )


class PaperWithS2Refs(Record):
    """PeerRead main paper where references have the full S2 data and their contexts."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[pr.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]
    references: Annotated[
        Sequence[S2Reference],
        Field(
            description="References from the paper with full S2 data and citation contexts."
        ),
    ]
    originality_rating: Annotated[
        int, Field(alias="rating", description="Stored originality rating")
    ]
    rationale: Annotated[str, Field(description="Rationale for the novelty rating")]
    year: Annotated[int | None, Field(description="Paper publication year")] = None
    publication_date: Annotated[
        date | None, Field(description="Paper publication date")
    ] = None
    arxiv_id: Annotated[str | None, Field(description="ID of the paper on arXiv")] = (
        None
    )

    @computed_field
    @cached_property
    def review(self) -> pr.PaperReview | None:
        """Get the review with median rating, breaking ties by confidence and rationale."""
        reviews = self.reviews
        if not reviews:
            return None

        sorted_reviews = sorted(reviews, key=lambda x: x.rating)
        median_idx = len(sorted_reviews) // 2
        median_rating = sorted_reviews[median_idx].rating
        median_reviews = [r for r in reviews if r.rating == median_rating]

        if len(median_reviews) == 1:
            return median_reviews[0]

        if reviews_with_confidence := [
            r for r in median_reviews if r.confidence is not None
        ]:
            return max(reviews_with_confidence, key=lambda x: x.confidence or 0)

        return min(median_reviews, key=lambda x: x.rationale)

    @computed_field
    @property
    def rating(self) -> int:
        """Recommendation rating from main review, scaled to 1-4.

        Uses the 'recommendation' field from other_ratings if available,
        otherwise falls back to the originality rating. Recommendation is
        on a 1-9 scale, scaled using non-linear binning via scale_recommendation().
        """
        if self.review is None:
            return self.originality_rating
        rec = self.review.other_ratings.get("recommendation")
        if rec is not None:
            return scale_recommendation(rec)
        return self.originality_rating

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @computed_field
    @property
    def id(self) -> str:
        """Identify an PeerRead by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)

    def main_text(self) -> str:
        """Join all paper sections to form the main text."""
        return clean_maintext("\n".join(s.text for s in self.sections))

    @classmethod
    def from_peer(cls, peer: pr.Paper, s2_references: Sequence[S2Reference]) -> Self:
        """Create paper with S2 references from original paper and S2 references."""
        return cls(
            title=peer.title,
            abstract=peer.abstract,
            reviews=peer.reviews,
            authors=peer.authors,
            sections=peer.sections,
            approval=peer.approval,
            conference=peer.conference,
            references=s2_references,
            originality_rating=peer.originality_rating,
            rationale=peer.rationale,
            year=peer.year,
            publication_date=peer.publication_date,
            arxiv_id=peer.arxiv_id,
        )


class PeerReadPaperWithS2(pr.Paper):
    """PeerRead paper with associated S2 paper information."""

    s2: PaperFromPeerRead
    fuzz_ratio: int

    @classmethod
    def from_peer(cls, pr: pr.Paper, s2_result: PaperFromPeerRead) -> Self:
        """Create new paper from an existing PeerRead paper and the S2 API result."""
        return cls(
            title=pr.title,
            abstract=pr.abstract,
            reviews=pr.reviews,
            authors=pr.authors,
            sections=pr.sections,
            approval=pr.approval,
            references=pr.references,
            conference=pr.conference,
            year=pr.year,
            publication_date=pr.publication_date,
            s2=s2_result,
            fuzz_ratio=title_ratio(pr.title, s2_result.title),
        )


class PaperWithRecommendations(Immutable):
    """PeerRead paper with a list of recommendations from the S2 API."""

    main_paper: PeerReadPaperWithS2
    recommendations: Sequence[Paper]


class PaperRecommended(Paper):
    """S2 paper recommended from PeerRead papers.

    Attributes:
        sources_peer: PeerRead titles for the the papers that led to this.
        sources_s2: S2 titles for the the papers that led to this.
    """

    sources_peer: Sequence[str]
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


class PeerReadWithFullS2(Record):
    """PeerRead main paper where references have the full S2 data."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[pr.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[PaperFromPeerRead],
        Field(description="References made in the paper with full S2 data"),
    ]

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @property
    def id(self) -> str:
        """Identify a PeerRead paper by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)


class PaperArea(Paper):
    """S2 paper with the areas that led to it."""

    areas: Sequence[str]


class ReferenceEnriched(pr.PaperReference):
    """PeerRead reference with the added data from the S2 API and the original S2 title.

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


class PaperWithReferenceEnriched(Immutable):
    """Paper from PeerRead where the references contain extra data from the S2 API."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[pr.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[ReferenceEnriched],
        Field(description="References made in the paper with their abstracts"),
    ]

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)


class ReferenceWithAbstract(pr.PaperReference):
    """PeerRead reference with the added abstract and the original S2 title.

    `s2title` is the title in the S2 data for the best match. It can be used to match
    back to the original S2 file if desired.
    """

    abstract: Annotated[str, Field(description="Abstract text")]
    s2title: Annotated[str, Field(description="Title from the S2 data")]
    paper_id: Annotated[str, Field(description="Paper ID in the S2 API")]

    @property
    def id(self) -> str:
        """Identify the reference by its S2 API ID."""
        return self.paper_id


class PaperWithFullReference(Record):
    """Paper from PeerRead where the references contain their abstract."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[pr.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[ReferenceWithAbstract],
        Field(description="References made in the paper with their abstracts"),
    ]

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)
