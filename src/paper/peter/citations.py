"""Build citations graph from PeerRead data with S2 papers and context polarities.

Input: `PaperWithContextClassfied`.
Output: `QueryResult` with lists of positive and negative `Citation`s.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from paper import embedding as emb
from paper.related_papers import ContextPolarity
from paper.util.serde import Record

logger = logging.getLogger(__name__)


class Graph(BaseModel):
    """Citation graph where paper citations are organised by title similarity."""

    model_config = ConfigDict(frozen=True)

    title_to_id: Mapping[str, str]
    """PeerRead paper titles to IDs."""
    id_polarity_to_cited: Mapping[str, Mapping[ContextPolarity, Sequence[Citation]]]
    """PeerRead paper IDs and context polarity to cited papers.

    Sorted by similarity score (descending) between paper and cited titles.
    """

    @classmethod
    def from_papers(
        cls,
        encoder: emb.Encoder,
        papers: Iterable[PaperWithContextClassfied],
        progress: bool = False,
    ) -> Self:
        """For each PeerRead paper, sort cited papers by title similarity.

        Cleans up the titles with `s2.clean_title`, then compares the PeerRead `title`
        with the S2 `title_peerread`.

        Args:
            encoder: Text to vector encoder to use on the nodes.
            papers: Papers to be processed into graph nodes.
            progress: If True, show a progress bar while generating node embeddings.
        """
        title_to_id: dict[str, str] = {}
        id_polarity_to_cited: dict[str, dict[ContextPolarity, list[Citation]]] = (
            defaultdict(dict)
        )

        logger.debug("Processing papers.")
        if progress:
            papers = tqdm(papers)

        for peer_paper in papers:
            title_to_id[peer_paper.title] = peer_paper.id
            peer_embedding = encoder.encode(_clean_title(peer_paper.title))

            s2_embeddings = encoder.encode([
                _clean_title(r.title_peer) for r in peer_paper.references
            ])
            s2_similarities = emb.similarities(peer_embedding, s2_embeddings)

            for polarity in ContextPolarity:
                id_polarity_to_cited[peer_paper.id][polarity] = [
                    Citation(
                        score=score,
                        title=paper.title,
                        abstract=paper.abstract,
                        paper_id=paper.id,
                        polarity=polarity,
                    )
                    for paper, score in sorted(
                        zip(peer_paper.references, s2_similarities),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    if paper.polarity.value == polarity.value
                ]

        logger.debug("Done processing papers.")

        return cls(id_polarity_to_cited=id_polarity_to_cited, title_to_id=title_to_id)

    def query_title(self, title: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from a PeerRead paper `title`.

        Note: prefer `query` for actual usage. See `title_to_id`.
        """
        return self.query(self.title_to_id[title], k)

    def query(self, paper_id: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from a PeerRead paper `id`.

        Returns `k` papers from each polarity.
        """
        if k == 0:
            return QueryResult(positive=[], negative=[])

        positive, negative = (
            self.id_polarity_to_cited[paper_id][polarity][:k]
            for polarity in (ContextPolarity.POSITIVE, ContextPolarity.NEGATIVE)
        )
        return QueryResult(positive=positive, negative=negative)


def _clean_title(title: str) -> str:
    """Clean-up title with a series of transformations.

    - Strip whitespace
    - Casefold (UTF-accurate lowercasing)
    - Remove all characters that are not alphanumeric or spaces
    - Collapse repeated spaces
    """
    title = title.strip().casefold()
    alpha_only = "".join(c for c in title if c.isalpha() or c.isspace())
    return " ".join(alpha_only.split())


class Citation(Record):
    """S2 paper cited by the PeerRead paper with the title similarity score and polarity."""

    score: float
    paper_id: str
    title: str
    abstract: str
    polarity: ContextPolarity

    @property
    def id(self) -> str:
        """Identify the Citation by its underlying paper ID."""
        return self.paper_id


class QueryResult(BaseModel):
    """Result of the citation query: the top K cited papers."""

    model_config = ConfigDict(frozen=True)

    positive: Sequence[Citation]
    negative: Sequence[Citation]


class PaperWithContextClassfied(Protocol):
    """Paper from PeerRead with each citation polarity classified.

    Input for the PETER Citations graph.
    """

    @property
    def title(self) -> str:
        """Paper title."""
        ...

    @property
    def id(self) -> str:
        """Unique identifier for the paper."""
        ...

    @property
    def references(self) -> Sequence[ReferenceClassified]:
        """References made in the paper with their polarity."""
        ...


class ReferenceClassified(Protocol):
    """Paper reference with a polarity."""

    @property
    def title_peer(self) -> str:
        """Paper title in the original PeerRead reference."""
        ...

    @property
    def title(self) -> str:
        """Paper title in the S2 API."""
        ...

    @property
    def id(self) -> str:
        """Unique identifier for the reference."""
        ...

    @property
    def abstract(self) -> str:
        """Abstract text."""
        ...

    @property
    def polarity(self) -> PolarityProto:
        """Reference polarity.

        Majority of all the context polarities. Ties are positive.
        """
        ...


class PolarityProto(Protocol):
    """Protocol representing an Enum with POSITIVE/NEGATIVE members.

    The goal is to decouple from the representation in other packages.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"

    @property
    def value(self) -> str:
        """Get value of the enum."""
        ...
