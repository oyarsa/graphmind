"""Build semantic graph based on backgrounds and targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict

from paper import embedding as emb
from paper.util.serde import Record


class Graph:
    """Semantic graph that finds other papers by background or target similarity."""

    _encoder: emb.Encoder
    """Text encoder used to convert backgrounds and targets to vectors."""
    _embeddings_background: emb.Matrix
    """Pre-computed embeddings for abstract backgrounds. Same order as `_backgrounds`"""
    _embeddings_target: emb.Matrix
    """Pre-computed embeddings for abstract targets Same order as `_targets`."""
    _background_to_paper: Mapping[str, PaperRelated]
    """Mapping from paper background to its paper."""
    _target_to_paper: Mapping[str, PaperRelated]
    """Mapping from paper target to its paper."""
    _backgrounds: Sequence[str]
    """Background nodes."""
    _targets: Sequence[str]
    """Target nodes."""

    def __init__(
        self,
        *,
        encoder: emb.Encoder,
        embeddings_background: emb.Matrix,
        embeddings_target: emb.Matrix,
        background_to_paper: Mapping[str, PaperRelated],
        target_to_paper: Mapping[str, PaperRelated],
        backgrounds: Sequence[str],
        targets: Sequence[str],
    ) -> None:
        self._encoder = encoder
        self._embeddings_background = embeddings_background
        self._embeddings_target = embeddings_target
        self._background_to_paper = background_to_paper
        self._target_to_paper = target_to_paper
        self._backgrounds = backgrounds
        self._targets = targets

    @classmethod
    def from_sentences(
        cls, encoder: emb.Encoder, papers: Iterable[PaperAnnotated]
    ) -> Self:
        """Build semantic graph from paper background and target."""
        background_to_paper = {
            paper.background: PaperRelated.from_ann(paper) for paper in papers
        }
        target_to_paper = {
            paper.target: PaperRelated.from_ann(paper) for paper in papers
        }

        backgrounds = sorted(background_to_paper)
        targets = sorted(target_to_paper)

        background_embs = encoder.batch_encode(backgrounds)
        target_embs = encoder.batch_encode(targets)

        return cls(
            encoder=encoder,
            embeddings_background=background_embs,
            embeddings_target=target_embs,
            background_to_paper=background_to_paper,
            target_to_paper=target_to_paper,
            backgrounds=backgrounds,
            targets=targets,
        )

    def query(self, paper: PaperAnnotated) -> QueryResult:
        """Get related papers by background and target."""

        bg_emb = self._encoder.encode(paper.background)
        bg_sim = emb.similarities(bg_emb, self._embeddings_background)
        bg_best = int(bg_sim.argmax())
        bg_node = self._backgrounds[bg_best]
        bg_score = bg_sim[bg_best]
        bg_result = self._background_to_paper[bg_node]

        tgt_emb = self._encoder.encode(paper.target)
        tgt_sim = emb.similarities(tgt_emb, self._embeddings_target)
        tgt_best = int(tgt_sim.argmax())
        tgt_node = self._targets[tgt_best]
        tgt_score = tgt_sim[tgt_best]
        tgt_result = self._target_to_paper[tgt_node]

        return QueryResult(
            backgrounds=[PaperResult.from_related(tgt_result, score=tgt_score)],
            targets=[PaperResult.from_related(bg_result, score=bg_score)],
        )

    def to_data(self) -> GraphData:
        """Convert graph to serialisable format."""
        return GraphData(
            background_to_paper=self._background_to_paper,
            target_to_paper=self._target_to_paper,
            backgrounds=self._backgrounds,
            targets=self._targets,
            encoder_model=self._encoder.model_name,
            embeddings_background=emb.MatrixData.from_matrix(
                self._embeddings_background
            ),
            embeddings_target=emb.MatrixData.from_matrix(self._embeddings_target),
        )


class PaperAnnotated(Protocol):
    """ASAP paper whose abstract has been separated into `background` and `target`."""

    @property
    def id(self) -> str:
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
    def background(self) -> str:
        """Background context from the paper: methods, algorithms, etc."""
        ...

    @property
    def target(self) -> str:
        """Target for the paper: datasets, problems, tasks, etc."""
        ...


class QueryResult(BaseModel):
    """Result from querying the semantic graph."""

    model_config = ConfigDict(frozen=True)

    targets: Sequence[PaperRelated]
    backgrounds: Sequence[PaperRelated]


class PaperRelated(Record):
    """Paper stored in the semantic graph."""

    title: str
    abstract: str
    paper_id: str

    @property
    def id(self) -> str:
        """Identify related paper by the underlying paper id."""
        return self.paper_id

    @classmethod
    def from_ann(cls, paper: PaperAnnotated) -> Self:
        """Construct related paper from input annotated paper.

        PaperAnnotated is an abstract/protocol type, and we need a concrete BaseModel.
        """
        return cls(title=paper.title, abstract=paper.abstract, paper_id=paper.id)


class PaperResult(PaperRelated):
    """Related papaer with similarity score."""

    score: float

    @classmethod
    def from_related(cls, paper: PaperRelated, *, score: float) -> Self:
        """Create result from paper and score."""
        return cls.model_validate(paper.model_dump() | {"score": score})


class GraphData(BaseModel):
    """Serialisation format for the semantic graph.

    It needs a separate object so it can properly serialise the embeddings.
    """

    model_config = ConfigDict(frozen=True)

    background_to_paper: Mapping[str, PaperRelated]
    """Mapping from paper background to its paper."""
    target_to_paper: Mapping[str, PaperRelated]
    """Mapping from paper target to its paper."""
    backgrounds: Sequence[str]
    """Background nodes."""
    targets: Sequence[str]
    """Target nodes."""
    encoder_model: str
    """Encoder used to generate the embeddings."""
    embeddings_background: emb.MatrixData
    """Embeddings for paper backgrounds. Same order as `backgrounds`."""
    embeddings_target: emb.MatrixData
    """Embeddings for paper targets. Same order as `targets`."""

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        """Initialise Semantic Graph from data object.

        Raises:
            ValueError: `encoder` model is different from the one that generated the
            graph.
        """
        if encoder.model_name != self.encoder_model:
            raise ValueError(
                f"Incompatible encoder. Expected '{self.encoder_model}', got"
                f" '{encoder.model_name}'"
            )
        return Graph(
            background_to_paper=self.background_to_paper,
            target_to_paper=self.target_to_paper,
            backgrounds=self.backgrounds,
            targets=self.targets,
            encoder=encoder,
            embeddings_background=self.embeddings_background.to_matrix(),
            embeddings_target=self.embeddings_target.to_matrix(),
        )
