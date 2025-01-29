"""Types and functions for managing related papers.

These papers can be retrived from graphs such as SciMON and PETER.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict

from paper import gpt
from paper.util.serde import Record


class PaperResult(Record):
    """PeerRead paper with its related papers queried from the PETER graph."""

    paper: gpt.PeerReadAnnotated
    results: QueryResult

    @property
    def id(self) -> str:
        """Identify graph result as the underlying paper's ID."""
        return self.paper.id


class QueryResult(BaseModel):
    """Combined query results from PETER graphs."""

    model_config = ConfigDict(frozen=True)

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


class PaperRelated(Record):
    """S2 paper cited by the PeerRead paper with the title similarity score and polarity."""

    source: PaperSource
    paper_id: str
    title: str
    abstract: str
    score: float
    polarity: ContextPolarity

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
