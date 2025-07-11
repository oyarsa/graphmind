"""Types and functions for managing related papers from citation and semantic graphs.

This module provides data structures and utilities for working with papers related to
a target paper through various relationships. It supports papers retrieved from
citation and semantic similarity graphs.

Key Concepts:
- Related Papers: Papers connected to a target paper through citations or semantic
  similarity.
- Query Results: Organised collections of related papers categorised by source and
  polarity.
- Polarity: Whether a relationship is positive (supportive) or negative (contrasting).
- Source: Papers can come from citation networks or semantic similarity searches.

The module handles:
1. Storage and retrieval of related papers with their metadata.
2. Deduplication of papers appearing in multiple result sets.
3. Priority-based conflict resolution when papers appear multiple times.
4. Integration with PETER graph queries for citation-based relationships.
5. Support for semantic similarity results from various sources.

Deduplication Priority (highest to lowest):
1. Citation-based papers over semantic similarity papers.
2. For semantic papers: background context over target context.
3. First occurrence wins for identical priority.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Self

from paper import gpt
from paper import peerread as pr
from paper.peerread import ContextPolarity
from paper.types import Immutable, PaperProxy, PaperSource
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper import peter


class PaperResult(Immutable, PaperProxy[gpt.PeerReadAnnotated]):
    """Container for a PeerRead paper with its related papers from citation graphs.

    Combines an annotated PeerRead paper with query results containing all related papers
    found through various graph-based searches. Acts as a proxy to the underlying paper
    while providing access to related papers.

    Attributes:
        paper: The annotated PeerRead paper being analysed.
        results: Combined query results containing all related papers.
    """

    paper: gpt.PeerReadAnnotated
    results: QueryResult


class QueryResult(Immutable):
    """Collection of related papers organised by source and polarity.

    Stores related papers in four categories based on how they were found (citation vs
    semantic) and their relationship polarity (positive vs negative). Provides utilities
    for accessing and deduplicating the complete set.

    Attributes:
        semantic_positive: Papers related by semantic similarity to target/contributions.
        semantic_negative: Papers related by semantic similarity to background/context.
        citations_positive: Papers cited positively (supportive references).
        citations_negative: Papers cited negatively (critical references).
    """

    semantic_positive: Sequence[PaperRelated]
    semantic_negative: Sequence[PaperRelated]
    citations_positive: Sequence[PaperRelated]
    citations_negative: Sequence[PaperRelated]

    @property
    def related(self) -> Iterable[PaperRelated]:
        """Iterate over all related papers regardless of source or polarity.

        Yields papers in order: positive citations, negative citations, positive semantic,
        negative semantic.
        """
        return itertools.chain(
            self.semantic_positive,
            self.semantic_negative,
            self.citations_positive,
            self.citations_negative,
        )

    def deduplicated(self) -> QueryResult:
        """Create a deduplicated version of query results using priority rules.

        When the same paper appears in multiple result sets, keeps only the highest
        priority occurrence. This ensures each paper appears exactly once while
        preserving the most important relationship type.

        Priority order (highest to lowest):
        1. Citation-based papers (both positive and negative).
        2. Semantic papers from background context (negative polarity).
        3. Semantic papers from target context (positive polarity).

        Returns:
            New QueryResult with each paper appearing at most once.
        """
        ps = PaperSource
        cp = ContextPolarity
        # Collect all papers with their original categories using enum-based keys
        all_papers = [
            *((p, (ps.CITATIONS, cp.POSITIVE)) for p in self.citations_positive),
            *((p, (ps.CITATIONS, cp.NEGATIVE)) for p in self.citations_negative),
            *((p, (ps.SEMANTIC, cp.NEGATIVE)) for p in self.semantic_negative),
            *((p, (ps.SEMANTIC, cp.POSITIVE)) for p in self.semantic_positive),
        ]

        # Deduplicate by paper_id, keeping highest priority
        seen_papers: dict[
            str, tuple[PaperRelated, tuple[PaperSource, ContextPolarity]]
        ] = {}
        for paper, category in all_papers:
            paper_id = paper.paper_id
            if paper_id not in seen_papers:
                seen_papers[paper_id] = (paper, category)
            else:
                existing_paper, _ = seen_papers[paper_id]
                # Keep higher priority paper
                if should_replace_paper(existing_paper, paper):
                    seen_papers[paper_id] = (paper, category)

        # Redistribute papers back to their categories
        new_citations_positive: list[PaperRelated] = []
        new_citations_negative: list[PaperRelated] = []
        new_semantic_positive: list[PaperRelated] = []
        new_semantic_negative: list[PaperRelated] = []

        for paper, (source, polarity) in seen_papers.values():
            if source == ps.CITATIONS and polarity == cp.POSITIVE:
                new_citations_positive.append(paper)
            elif source == ps.CITATIONS and polarity == cp.NEGATIVE:
                new_citations_negative.append(paper)
            elif source == ps.SEMANTIC and polarity == cp.POSITIVE:
                new_semantic_positive.append(paper)
            elif source == ps.SEMANTIC and polarity == cp.NEGATIVE:
                new_semantic_negative.append(paper)

        return QueryResult(
            semantic_positive=new_semantic_positive,
            semantic_negative=new_semantic_negative,
            citations_positive=new_citations_positive,
            citations_negative=new_citations_negative,
        )


def should_replace_paper(existing: PaperRelated, new: PaperRelated) -> bool:
    """Determine if new paper should replace existing paper based on priority.

    Used during deduplication to decide which occurrence to keep when the same paper
    appears multiple times. Implements the priority rules for paper source and polarity
    comparisons.

    Args:
        existing: Currently stored paper.
        new: Candidate paper that might replace it.

    Returns:
        True if new paper has higher priority and should replace existing.
    """
    # Citations always have priority over semantic
    if existing.source == PaperSource.CITATIONS and new.source == PaperSource.SEMANTIC:
        return False
    if existing.source == PaperSource.SEMANTIC and new.source == PaperSource.CITATIONS:
        return True

    # Both are same source type
    if existing.source == PaperSource.SEMANTIC and new.source == PaperSource.SEMANTIC:
        # For semantic papers: background (negative) > target (positive)
        if (
            existing.polarity == ContextPolarity.NEGATIVE
            and new.polarity == ContextPolarity.POSITIVE
        ):
            return False
        if (
            existing.polarity == ContextPolarity.POSITIVE
            and new.polarity == ContextPolarity.NEGATIVE
        ):
            return True

    # If same source and polarity, keep existing (first one wins)
    return False


class PaperRelated(Record):
    """Related paper with metadata about its relationship to the target paper.

    Represents a paper that is related to the target paper through either citation or
    semantic similarity. Includes relationship metadata like similarity scores, polarity,
    and context information.

    Attributes:
        source: How this paper was found (citations or semantic search).
        paper_id: Unique Semantic Scholar paper identifier.
        title: Paper title.
        abstract: Paper abstract.
        score: Similarity/relevance score (interpretation depends on source).
        polarity: Positive (supportive) or negative (contrasting) relationship.
        year: Year of publication.
        authors: List of author names.
        venue: Publication venue name.
        citation_count: Number of papers that cite this paper.
        reference_count: Number of papers this paper references.
        influential_citation_count: Number of influential citations.
        corpus_id: Semantic Scholar's secondary identifier.
        url: URL to paper on Semantic Scholar website.
        arxiv_id: arXiv identifier if available.
        contexts: Citation contexts (for citation-based papers).
        background: Background text that matched (for semantic papers).
        target: Target text that matched (for semantic papers).
    """

    source: PaperSource
    paper_id: str
    title: str
    abstract: str
    score: float
    polarity: ContextPolarity

    year: int | None = None
    authors: Sequence[str] | None = None
    venue: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    corpus_id: int | None = None
    url: str | None = None
    arxiv_id: str | None = None

    contexts: Sequence[pr.CitationContext] | None = None  # For citation-based papers
    background: str | None = None  # For semantic papers matched by background
    target: str | None = None  # For semantic papers matched by target

    @property
    def id(self) -> str:
        """Get the unique identifier for this related paper.

        Returns the Semantic Scholar paper ID, ensuring consistent identification across
        different sources and result sets.
        """
        return self.paper_id

    @classmethod
    def from_(
        cls,
        paper: peter.Citation | peter.SemanticResult,
        *,
        source: PaperSource,
        polarity: ContextPolarity,
        contexts: Sequence[pr.CitationContext] | None = None,
        background: str | None = None,
        target: str | None = None,
    ) -> Self:
        """Factory method to create PaperRelated from various paper data types.

        Converts abstract paper representations (like Citation or SemanticResult) into
        concrete PaperRelated instances with full metadata. Supports both citation-based
        and semantic similarity-based paper relationships.

        Args:
            paper: Source paper data (Citation or SemanticResult).
            source: How this paper was found.
            polarity: Relationship polarity.
            contexts: Citation contexts (for citation papers).
            background: Matching background text (for semantic papers).
            target: Matching target text (for semantic papers).

        Returns:
            New PaperRelated instance with all metadata.
        """
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            score=paper.score,
            source=source,
            polarity=polarity,
            year=paper.year,
            authors=paper.authors,
            venue=paper.venue,
            citation_count=paper.citation_count,
            reference_count=paper.reference_count,
            influential_citation_count=paper.influential_citation_count,
            corpus_id=paper.corpus_id,
            url=paper.url,
            arxiv_id=paper.arxiv_id,
            contexts=contexts,
            background=background,
            target=target,
        )
