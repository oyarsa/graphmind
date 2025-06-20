"""Types and functions for managing related papers.

These papers can be retrieved from graphs such as SciMON and PETER.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Protocol, Self

from paper import gpt
from paper import peerread as pr
from paper.types import Immutable, PaperProxy
from paper.util.serde import Record


class PaperResult(Immutable, PaperProxy[gpt.PeerReadAnnotated]):
    """PeerRead paper with its related papers queried from the PETER graph."""

    paper: gpt.PeerReadAnnotated
    results: QueryResult


class QueryResult(Immutable):
    """Combined query results from PETER graphs."""

    semantic_positive: Sequence[PaperRelated]
    semantic_negative: Sequence[PaperRelated]
    citations_positive: Sequence[PaperRelated]
    citations_negative: Sequence[PaperRelated]

    @property
    def related(self) -> Iterable[PaperRelated]:
        """Retrieve all related papers from all polarities."""
        return itertools.chain(
            self.semantic_positive,
            self.semantic_negative,
            self.citations_positive,
            self.citations_negative,
        )

    def deduplicated(self) -> QueryResult:
        """Return a new QueryResult with deduplicated papers using priority rules.

        Priority order:
        1. Citations > Semantic
        2. For semantic: Background (negative polarity) > Target (positive polarity)
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
                if _should_replace_paper(existing_paper, paper):
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


def _should_replace_paper(existing: PaperRelated, new: PaperRelated) -> bool:
    """Determine if new paper should replace existing paper based on priority.

    Priority order:
    1. Citations > Semantic
    2. For semantic: Background (negative polarity) > Target (positive polarity)
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
    """S2 paper cited by the PeerRead paper with the title similarity score and polarity."""

    source: PaperSource
    paper_id: str
    title: str
    abstract: str
    score: float
    polarity: ContextPolarity

    contexts: Sequence[pr.CitationContext] | None = None  # For citation-based papers
    background: str | None = None  # For semantic papers matched by background
    target: str | None = None  # For semantic papers matched by target

    @property
    def id(self) -> str:
        """Identify the Citation by its underlying paper ID."""
        return self.paper_id

    @classmethod
    def from_(
        cls, paper: RelatedResult, *, source: PaperSource, polarity: ContextPolarity
    ) -> Self:
        """Create concrete paper result from abstract/protocol data and a source."""
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            score=paper.score,
            source=source,
            polarity=polarity,
        )


class PaperSource(StrEnum):
    """Denote where the related paper came from."""

    CITATIONS = "citations"
    SEMANTIC = "semantic"


class ContextPolarity(StrEnum):
    """Citation enum for polarity."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class RelatedResult(Protocol):
    """Protocol for retrieved related papers used to build a concrete result."""

    @property
    def paper_id(self) -> str:
        """Paper unique identifier."""
        ...

    @property
    def title(self) -> str:
        """Paper title."""
        ...

    @property
    def abstract(self) -> str:
        """Paper abstract."""
        ...

    @property
    def score(self) -> float:
        """Paper similarity score."""
        ...
