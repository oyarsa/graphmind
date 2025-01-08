"""Data models for types returned by the Semantic Scholar API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self, override

from pydantic import BaseModel, ConfigDict, Field

from paper import peerread
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
    ]
    authors: Annotated[Sequence[Author] | None, Field(description="Paper authors")]

    @property
    def id(self) -> str:
        """Identify paper by the S2 API paper ID."""
        return self.paper_id


class S2Reference(PaperFromPeerRead):
    """S2 paper as a reference with the original contexts."""

    contexts: Sequence[peerread.CitationContext]

    @classmethod
    def from_(
        cls, paper: PaperFromPeerRead, *, contexts: Sequence[peerread.CitationContext]
    ) -> Self:
        """Create new instance by copying data from S2Paper, in addition to the contexts."""
        return cls.model_validate(paper.model_dump() | {"contexts": contexts})


class PaperWithS2Refs(Record):
    """PeerRead main paper where references have the full S2 data and their contexts."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[peerread.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[peerread.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[S2Reference],
        Field(
            description="References from the paper with full S2 data and citation contexts."
        ),
    ]

    @property
    def id(self) -> str:
        """Identify an PeerRead by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)


class PeerReadPaperWithS2(peerread.Paper):
    """PeerRead paper with associated S2 paper information."""

    s2: PaperFromPeerRead
    fuzz_ratio: int

    @classmethod
    def from_peer(cls, pr: peerread.Paper, s2_result: PaperFromPeerRead) -> Self:
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
            s2=s2_result,
            fuzz_ratio=title_ratio(pr.title, s2_result.title),
        )


class PaperWithRecommendations(BaseModel):
    """PeerRead paper with a list of recommendations from the S2 API."""

    model_config = ConfigDict(frozen=True)

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
        Sequence[peerread.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[peerread.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[PaperFromPeerRead],
        Field(description="References made in the paper with full S2 data"),
    ]

    @property
    def id(self) -> str:
        """Identify a PeerRead paper by the combination of its `title` and `abstract`.

        The `title` isn't unique by itself, but `title+abstract` is. Instead of passing
        full text around, I hash it.
        """
        return hashstr(self.title + self.abstract)


class PaperArea(Paper):
    """S2 paper with the areas that led to it."""

    model_config = ConfigDict(frozen=True)

    areas: Sequence[str]


class ReferenceEnriched(peerread.PaperReference):
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


class PaperWithReferenceEnriched(BaseModel):
    """Paper from PeerRead where the references contain extra data from the S2 API."""

    model_config = ConfigDict(frozen=True)

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[peerread.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[peerread.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[ReferenceEnriched],
        Field(description="References made in the paper with their abstracts"),
    ]


class ReferenceWithAbstract(peerread.PaperReference):
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
        Sequence[peerread.PaperReview], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[peerread.PaperSection], Field(description="Sections in the paper text")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    rationale: Annotated[str, Field(description="Rationale for novelty rating")]
    references: Annotated[
        Sequence[ReferenceWithAbstract],
        Field(description="References made in the paper with their abstracts"),
    ]

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)
