"""Build semantic graph based on backgrounds and targets.

Input: `PaperAnnotated`.
Output: `QueryResult` with `targets` and `backgrounds`-related `PaperResult`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from paper import embedding as emb
from paper import semantic_scholar as s2
from paper.gpt.openai_encoder import OpenAIEncoderSync
from paper.types import Immutable, Matrix
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper import gpt

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class _Components:
    """Elements of a graph with nodes, node embeddings and node to data mapping."""

    nodes: Sequence[str]
    """Sentence nodes."""
    embeddings: Matrix
    """Pre-computed embeddings for nodes. Same order as `nodes`."""
    node_to_paper: Mapping[str, _PaperRelated]
    """Mapping from node to its original paper."""

    @classmethod
    def from_node_paper(
        cls,
        encoder: OpenAIEncoderSync,
        node_to_paper: Mapping[str, _PaperRelated],
        *,
        progress: bool = False,
    ) -> Self:
        """Create component from mapping of node to paper with node embeddings."""
        nodes = sorted(node_to_paper)
        embeddings = encoder.batch_encode(nodes, progress=progress)
        return cls(embeddings=embeddings, nodes=nodes, node_to_paper=node_to_paper)

    def to_data(self) -> _ComponentsData:
        """Convert components to serialisable format."""
        return _ComponentsData(
            embeddings=emb.MatrixData.from_matrix(self.embeddings),
            node_to_paper=self.node_to_paper,
            nodes=self.nodes,
        )


class _ComponentsData(Immutable):
    """Serialisation format for `_Components`.

    This is needed to convert the embedding matrix to JSON.
    """

    embeddings: emb.MatrixData
    node_to_paper: Mapping[str, _PaperRelated]
    nodes: Sequence[str]

    def to_components(self) -> _Components:
        """Deserialise into a real `_Components` object."""
        return _Components(
            embeddings=self.embeddings.to_matrix(),
            node_to_paper=self.node_to_paper,
            nodes=self.nodes,
        )


class Graph:
    """Semantic graph that finds other papers by background or target similarity.

    - Target: methods and objectives.
    - Background: context, problem and motivation.
    """

    _encoder: OpenAIEncoderSync
    """Text encoder used to convert backgrounds and targets to vectors."""
    _targets: _Components
    """Target nodes: methods and objectives."""
    _backgrounds: _Components
    """Background nodes: context, problem and motivation."""

    def __init__(
        self,
        *,
        encoder: OpenAIEncoderSync,
        targets: _Components,
        backgrounds: _Components,
    ) -> None:
        self._encoder = encoder
        self._backgrounds = backgrounds
        self._targets = targets

    @classmethod
    def from_papers(
        cls,
        encoder: OpenAIEncoderSync,
        papers: Iterable[gpt.PaperAnnotated],
        *,
        progress: bool = False,
    ) -> Self:
        """Build semantic graph from paper backgrounds and targets.

        Args:
            encoder: Text to vector encoder to use on the nodes.
            papers: Papers to be processed into graph nodes.
            progress: If True, show a progress bar while generating node embeddings.
        """
        background_to_paper = {
            paper.background: _PaperRelated.from_ann(paper) for paper in papers
        }
        target_to_paper = {
            paper.target: _PaperRelated.from_ann(paper) for paper in papers
        }

        return cls(
            encoder=encoder,
            backgrounds=_Components.from_node_paper(
                encoder, background_to_paper, progress=progress
            ),
            targets=_Components.from_node_paper(
                encoder, target_to_paper, progress=progress
            ),
        )

    def query(self, background: str, target: str, k: int) -> QueryResult:
        """Get top K related papers by background and target.

        This gets top K for each side, sorted by similarity between sides (i.e. `paper`
        background vs the graph papers' backgrounds).

        Note that some papers may appear in both retrieved lists. Having doubled papers
        is not informative for our goal, so we remove those before taking the top K.

        Implementation note: we take the top 2K, filter out doubled papers, then take
        top K of the result. It's theoretically possible that this would yield less than
        K papers, but it's unlikely.
        """
        if k == 0:
            return QueryResult(backgrounds=[], targets=[])

        # Take top 2K because we'll remove some items next.
        matches_background = self._query(background, self._backgrounds, k=2 * k)
        matches_target = self._query(target, self._targets, k=2 * k)
        logger.debug("Background matches: %d.", len(matches_background))
        logger.debug("Target matches: %d.", len(matches_target))

        # Remove papers that appear in both lists
        ids_background = {p.paper_id for p in matches_background}
        ids_target = {p.paper_id for p in matches_target}
        ids_common = ids_background & ids_target
        logger.debug("Common papers: %d.", len(ids_common))

        filtered_background = [p for p in matches_background if p.id not in ids_common]
        filtered_target = [p for p in matches_target if p.id not in ids_common]
        logger.debug("Background filtered: %d.", len(filtered_background))
        logger.debug("Target filtered: %d.", len(filtered_target))

        return QueryResult(
            backgrounds=filtered_background[:k], targets=filtered_target[:k]
        )

    def query_threshold(
        self, background: str, target: str, threshold: float, retrieved_k: int = 100
    ) -> QueryResult:
        """Get semantic-related with score above `threshold`.

        First, we fetch `retrieved_k` items, then gets only the results above the
        threshold.
        """
        results = self.query(background, target, retrieved_k)
        return QueryResult(
            backgrounds=[b for b in results.backgrounds if b.score >= threshold],
            targets=[t for t in results.backgrounds if t.score >= threshold],
        )

    def _query(
        self, sentence: str, elements: _Components, *, k: int
    ) -> list[SemanticResult]:
        """Get top K nodes in `elements` by similarity with `sentence`.

        Results are sorted by their scores, descending.
        """
        embedding = self._encoder.encode(sentence)
        sim = emb.similarities(embedding, elements.embeddings)

        return [
            SemanticResult.from_related(
                related=elements.node_to_paper[elements.nodes[idx]], score=sim[idx]
            )
            for idx in emb.top_k_indices(sim, k)
        ]

    def to_data(self) -> GraphData:
        """Convert graph to serialisable format."""
        return GraphData(
            backgrounds=self._backgrounds.to_data(),
            targets=self._targets.to_data(),
            encoder_model=self._encoder.model_name,
        )


class QueryResult(Immutable):
    """Result from querying the semantic graph."""

    targets: Sequence[SemanticResult]
    """Top K similar papers by target sentence similarity."""
    backgrounds: Sequence[SemanticResult]
    """Top K similar papers by background sentence similarity."""


class _PaperRelated(Record):
    """Paper stored in the semantic graph."""

    title: str
    abstract: str
    paper_id: str
    background: str
    target: str
    year: int | None = None
    authors: Sequence[str] | None = None
    venue: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    corpus_id: int | None = None
    url: str | None = None
    arxiv_id: str | None = None

    @property
    def id(self) -> str:
        """Identify related paper by the underlying paper id."""
        return self.paper_id

    @classmethod
    def from_ann(cls, paper: gpt.PaperAnnotated) -> Self:
        """Construct related paper from input annotated paper."""
        match p := paper.paper:
            case s2.Paper():
                authors = [a.name for a in p.authors or [] if a.name]
                conference = None
                arxiv_id = None
                # Extract S2 metadata
                citation_count = p.citation_count
                reference_count = p.reference_count
                influential_citation_count = p.influential_citation_count
                corpus_id = p.corpus_id
                url = p.url
            case s2.PaperWithS2Refs():
                authors = p.authors
                conference = p.conference
                arxiv_id = p.arxiv_id
                # PeerRead papers don't have these fields
                citation_count = None
                reference_count = None
                influential_citation_count = None
                corpus_id = None
                url = None

        return cls(
            title=paper.title,
            abstract=paper.abstract,
            paper_id=paper.id,
            year=paper.paper.year,
            authors=authors,
            venue=conference,
            citation_count=citation_count,
            reference_count=reference_count,
            influential_citation_count=influential_citation_count,
            corpus_id=corpus_id,
            url=url,
            arxiv_id=arxiv_id,
            background=paper.background,
            target=paper.target,
        )


class SemanticResult(_PaperRelated):
    """Related paper with similarity score."""

    score: float

    @classmethod
    def from_related(cls, *, related: _PaperRelated, score: float) -> Self:
        """Create result from base related paper and score."""
        return cls.model_validate(related.model_dump() | {"score": score})


class GraphData(Immutable):
    """Serialisation format for the semantic graph.

    It needs a separate object so it can properly serialise the embeddings.
    """

    targets: _ComponentsData
    """Target nodes."""
    backgrounds: _ComponentsData
    """Background nodes."""
    encoder_model: str
    """Name of the encoder model used to generate the embeddings."""

    def to_graph(self, encoder: OpenAIEncoderSync | None = None) -> Graph:
        """Initialise Semantic graph from serialised object.

        Raises:
            ValueError: `encoder` model is different from the one that generated the
            graph.
        """
        if encoder is None:
            encoder = OpenAIEncoderSync(self.encoder_model)
        if encoder.model_name != self.encoder_model:
            raise ValueError(
                f"Incompatible encoder model. Expected '{self.encoder_model}', got"
                f" '{encoder.model_name}'."
            )

        return Graph(
            backgrounds=self.backgrounds.to_components(),
            targets=self.targets.to_components(),
            encoder=encoder,
        )
