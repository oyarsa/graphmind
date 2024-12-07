"""Full graph containing citation and semantic subgraphs."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict

from paper import embedding as emb
from paper.peter import citations, semantic
from paper.util import Timer
from paper.util.serde import Record, load_data

logger = logging.getLogger(__name__)


class Graph:
    """Graph to retrieve positive and negative related papers.

    Uses citation and semantic subgraphs to find those related papers:
    - Negative papers: those whose citation contexts are negative, or whose goal is the
      same but the methods are different.
    - Positive papers: those that positive citation contexts or share the same method
      but have different goals.
    """

    CITATION_TOP_K: ClassVar[int] = 5
    SEMANTIC_TOP_K: ClassVar[int] = 5

    _citation: citations.Graph
    _semantic: semantic.Graph
    _encoder_model: str

    def __init__(
        self, citation: citations.Graph, semantic: semantic.Graph, encoder_model: str
    ) -> None:
        self._citation = citation
        self._semantic = semantic
        self._encoder_model = encoder_model

    def to_data(self) -> GraphData:
        """Convert `Graph` object to a serialisable format."""
        return GraphData(
            semantic=self._semantic.to_data(),
            citation=self._citation,
            encoder_model=self._encoder_model,
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
        papers_semantic = self._semantic.query(background, target, k=semantic_k)
        papers_citation = self._citation.query(paper_id, k=citation_k)

        return QueryResult(
            semantic_positive=[
                PaperRelated.from_(p, source=PaperSource.SEMANTIC)
                for p in papers_semantic.targets
            ],
            semantic_negative=[
                PaperRelated.from_(p, source=PaperSource.SEMANTIC)
                for p in papers_semantic.backgrounds
            ],
            citations_positive=[
                PaperRelated.from_(p, source=PaperSource.CITATIONS)
                for p in papers_citation.positive
            ],
            citations_negative=[
                PaperRelated.from_(p, source=PaperSource.CITATIONS)
                for p in papers_citation.negative
            ],
        )

    @classmethod
    def from_papers(
        cls,
        encoder: emb.Encoder,
        papers_ann: Iterable[semantic.PaperAnnotated],
        papers_context: Iterable[citations.PaperWithContextClassfied],
    ) -> Self:
        """Create new PETER graph from annotated papers and classified contexts."""

        logger.debug("Creating semantic graph.")
        with Timer("Semantic") as timer_semantic:
            semantic_graph = semantic.Graph.from_papers(
                encoder, papers_ann, progress=True
            )
        logger.debug(timer_semantic)

        logger.debug("Creating citations graph.")
        with Timer("Citations") as timer_citations:
            citation_graph = citations.Graph.from_papers(
                encoder, papers_context, progress=True
            )
        logger.debug(timer_citations)

        return cls(
            citation=citation_graph,
            semantic=semantic_graph,
            encoder_model=encoder.model_name,
        )


class QueryResult(BaseModel):
    """Combined query results from PETER graphs."""

    model_config = ConfigDict(frozen=True)

    semantic_positive: Sequence[PaperRelated]
    semantic_negative: Sequence[PaperRelated]
    citations_positive: Sequence[PaperRelated]
    citations_negative: Sequence[PaperRelated]

    @property
    def positive(self) -> Sequence[PaperRelated]:
        """Retrieve all positive papers."""
        return list(itertools.chain(self.semantic_positive, self.citations_positive))

    @property
    def negative(self) -> Sequence[PaperRelated]:
        """Retrieve all negative papers."""
        return list(itertools.chain(self.semantic_negative, self.citations_negative))


class PaperRelated(Record):
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
