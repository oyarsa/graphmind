"""Full graph containing citation and semantic subgraphs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict

from paper.peter import citations, semantic
from paper.util.serde import Record, load_data


@dataclass(frozen=True, kw_only=True)
class Graph:
    """Graph with citation and semantic subgraphs."""

    CITATION_TOP_K: ClassVar[int] = 5
    SEMANTIC_TOP_K: ClassVar[int] = 5

    citation: citations.Graph
    semantic: semantic.Graph
    encoder_model: str

    def to_data(self) -> GraphData:
        """Convert `Graph` object to a serialisable format."""
        return GraphData(
            semantic=self.semantic.to_data(),
            citation=self.citation,
            encoder_model=self.encoder_model,
        )

    def query_all(
        self,
        paper_id: str,
        background: str,
        target: str,
        *,
        semantic_k: int = SEMANTIC_TOP_K,
        citation_k: int = CITATION_TOP_K,
    ) -> QueryResult:
        """Find papers related to `paper` through citations and semantic similarity."""
        papers_semantic = self.semantic.query(background, target, k=semantic_k)
        papers_citation = self.citation.query(paper_id, k=citation_k)

        return QueryResult(
            semantic_positive=[
                PaperResult.from_(p, source=PaperSource.SEMANTIC)
                for p in papers_semantic.targets
            ],
            semantic_negative=[
                PaperResult.from_(p, source=PaperSource.SEMANTIC)
                for p in papers_semantic.backgrounds
            ],
            citations_positive=[
                PaperResult.from_(p, source=PaperSource.CITATIONS)
                for p in papers_citation.positive
            ],
            citations_negative=[
                PaperResult.from_(p, source=PaperSource.CITATIONS)
                for p in papers_citation.negative
            ],
        )


class QueryResult(BaseModel):
    """Combined query results from PETER graphs."""

    model_config = ConfigDict(frozen=True)

    semantic_positive: Sequence[PaperResult]
    semantic_negative: Sequence[PaperResult]
    citations_positive: Sequence[PaperResult]
    citations_negative: Sequence[PaperResult]


class PaperResult(Record):
    """S2 paper cited by the ASAP paper with the title similarity score and polarity."""

    source: PaperSource
    paper_id: str
    title: str
    abstract: str
    score: float

    @property
    def id(self) -> str:
        """Identify the Citation by its underlying paper ID."""
        return self.paper_id

    @classmethod
    def from_(cls, paper: _PaperResult, *, source: PaperSource) -> Self:
        """Create concrete paper result from abstract/protocol data and a source."""
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            score=paper.score,
            source=source,
        )


class PaperSource(StrEnum):
    """Denote where the related paper came from."""

    CITATIONS = "citations"
    SEMANTIC = "semantic"


class _PaperResult(Protocol):
    """Protocol for papers used to build a concrete result."""

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


class GraphData(BaseModel):
    """Serialisation format for `Graph`."""

    model_config = ConfigDict(frozen=True)

    citation: citations.Graph
    semantic: semantic.GraphData
    encoder_model: str

    def to_graph(self) -> Graph:
        """Create full `Graph` object from serialised data."""
        return Graph(
            citation=self.citation,
            semantic=self.semantic.to_graph(),
            encoder_model=self.encoder_model,
        )


def graph_from_json(file: Path) -> Graph:
    """Read full `Graph` from a JSON `file`.

    See `GraphData` for more info.
    """
    return load_data(file, GraphData, single=True).to_graph()
