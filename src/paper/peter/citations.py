"""Build citations graph from PeerRead data with S2 papers and context polarities.

Input: `PaperWithContextClassfied`.
Output: `QueryResult` with lists of positive and negative `Citation`s.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Self

from tqdm import tqdm

from paper import embedding as emb
from paper import peerread as pr
from paper.peerread import ContextPolarity
from paper.types import Immutable
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper.gpt.classify_contexts import PaperWithContextClassfied

logger = logging.getLogger(__name__)


class Graph(Immutable):
    """Citation graph where paper citations are organised by title similarity."""

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

            s2_embeddings = encoder.encode_multi([
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
                        contexts=[
                            pr.CitationContext(
                                sentence=ctx.text,
                                polarity=ctx.prediction,
                            )
                            for ctx in paper.contexts
                            if ctx.prediction == polarity
                        ],
                    )
                    for paper, score in sorted(
                        zip(peer_paper.references, s2_similarities),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    if paper.polarity == polarity
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

    def query_threshold(self, paper_id: str, threshold: float) -> QueryResult:
        """Get top `k` cited papers by title similarity from a PeerRead paper `id`.

        Returns `k` papers from each polarity.
        """
        positive, negative = (
            self.id_polarity_to_cited[paper_id][polarity]
            for polarity in (ContextPolarity.POSITIVE, ContextPolarity.NEGATIVE)
        )
        return QueryResult(
            positive=[p for p in positive if p.score >= threshold],
            negative=[n for n in negative if n.score >= threshold],
        )


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
    contexts: Sequence[pr.CitationContext] | None = None

    @property
    def id(self) -> str:
        """Identify the Citation by its underlying paper ID."""
        return self.paper_id


class QueryResult(Immutable):
    """Result of the citation query: the top K cited papers."""

    positive: Sequence[Citation]
    negative: Sequence[Citation]
