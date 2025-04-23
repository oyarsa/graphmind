"""Full graph containing citation and semantic subgraphs."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict

from paper import embedding as emb
from paper.peter import citations, semantic
from paper.peter.citations import ContextPolarity
from paper.related_papers import PaperRelated, PaperSource, QueryResult
from paper.util import Timer
from paper.util.serde import load_data_single, save_data

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

    CITATION_FILENAME: ClassVar[str] = "citation_graph.json"
    SEMANTIC_FILENAME: ClassVar[str] = "semantic_graph.json"
    METADATA_FILENAME: ClassVar[str] = "metadata.json"

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

    def query_threshold(
        self,
        paper_id: str,
        background: str,
        target: str,
        semantic_threshold: float,
        citation_threshold: float,
    ) -> QueryResult:
        """Find papers related to `paper` through citations and semantic similarity."""
        papers_semantic = self._semantic.query_threshold(
            background, target, semantic_threshold
        )
        papers_citation = self._citation.query_threshold(paper_id, citation_threshold)

        return _result_from_related(papers_semantic, papers_citation)

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
        return _result_from_related(papers_semantic, papers_citation)

    @classmethod
    def build(
        cls,
        encoder: emb.Encoder,
        papers_ann: Iterable[semantic.PaperAnnotated],
        papers_context: Iterable[citations.PaperWithContextClassfied],
        output_dir: Path,
    ) -> None:
        """Build PETER graph from annotated papers and classified contexts.

        Args:
            encoder: Text to vector encoder to use on the nodes.
            papers_ann: Papers to be processed into semantic graph nodes.
            papers_context: Papers to be processed into citation graph nodes.
            output_dir: Directory where to save the graph.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        citation_file = output_dir / cls.CITATION_FILENAME
        semantic_file = output_dir / cls.SEMANTIC_FILENAME
        metadata_file = output_dir / cls.METADATA_FILENAME

        logger.debug("Creating semantic graph.")
        with Timer("Semantic") as timer_semantic:
            semantic_graph = semantic.Graph.from_papers(
                encoder, papers_ann, progress=True
            )
            logger.debug("Saving semantic graph")
            save_data(semantic_file, semantic_graph.to_data())
        logger.debug(timer_semantic)

        # Clear the semantic graph to free memory
        del semantic_graph
        # Collect twice to handle cycles
        gc.collect()
        gc.collect()

        logger.debug("Creating citations graph.")
        with Timer("Citations") as timer_citations:
            citation_graph = citations.Graph.from_papers(
                encoder, papers_context, progress=True
            )
            logger.debug("Saving citation graph")
            save_data(citation_file, citation_graph)
        logger.debug(timer_citations)

        logger.debug("Saving graph metadata")
        metadata_file.write_text(json.dumps({"encoder_model": encoder.model_name}))

    @classmethod
    def load(cls, graph_dir: Path) -> Self:
        """Load a Graph from a directory.

        Raises:
            pydantic.ValidationError: If the graph data is malformed.
            FileNotFoundError: If any of the required files is missing.
        """
        citation_file = graph_dir / cls.CITATION_FILENAME
        semantic_file = graph_dir / cls.SEMANTIC_FILENAME
        metadata_file = graph_dir / cls.METADATA_FILENAME

        if not all(f.exists() for f in [citation_file, semantic_file, metadata_file]):
            raise FileNotFoundError(f"Missing one or more graph files in {graph_dir}")

        metadata: dict[str, str] = json.loads(metadata_file.read_bytes())
        encoder_model = metadata["encoder_model"]

        citation_graph = load_data_single(citation_file, citations.Graph)

        semantic_data = load_data_single(semantic_file, semantic.GraphData)
        semantic_graph = semantic_data.to_graph(emb.Encoder(encoder_model))

        return cls(
            citation=citation_graph,
            semantic=semantic_graph,
            encoder_model=encoder_model,
        )


def _result_from_related(
    papers_semantic: semantic.QueryResult, papers_citation: citations.QueryResult
) -> QueryResult:
    return QueryResult(
        semantic_positive=[
            PaperRelated.from_(
                p, source=PaperSource.SEMANTIC, polarity=ContextPolarity.POSITIVE
            )
            for p in papers_semantic.targets
        ],
        semantic_negative=[
            PaperRelated.from_(
                p, source=PaperSource.SEMANTIC, polarity=ContextPolarity.NEGATIVE
            )
            for p in papers_semantic.backgrounds
        ],
        citations_positive=[
            PaperRelated.from_(
                p, source=PaperSource.CITATIONS, polarity=ContextPolarity.POSITIVE
            )
            for p in papers_citation.positive
        ],
        citations_negative=[
            PaperRelated.from_(
                p, source=PaperSource.CITATIONS, polarity=ContextPolarity.NEGATIVE
            )
            for p in papers_citation.negative
        ],
    )


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
