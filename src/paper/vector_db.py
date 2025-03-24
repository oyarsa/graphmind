"""Vector database for retrieving similar sentences by cosine similarity.

Uses SentenceTransformers to build sentence embeddings and faiss to create the index.
"""
# pyright: basic

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

import faiss  # type: ignore
import typer
from pydantic import BaseModel, ConfigDict

from paper import embedding as emb
from paper.embedding import DEFAULT_SENTENCE_MODEL

DEFAULT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.6

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


class SearchMatch(BaseModel):
    """Match from searching a query in the database."""

    model_config = ConfigDict(frozen=True)

    sentence: str
    """Sentence retrieved from the database."""
    score: float
    """Similarity score with the query."""


class SearchResult(BaseModel):
    """Search result with all matches for a query."""

    model_config = ConfigDict(frozen=True)

    query: str
    """Sentence used to query the database."""
    matches: list[SearchMatch]
    """Retrieved sentences with scores."""


class VectorDatabase:
    """Database for embedded sentences."""

    INDEX_FILE = "index.faiss"
    """File with indexed documents with vector embeddings."""
    METADATA_FILE = "metadata.json"
    """File with JSON metadata needed to construct the database."""

    encoder: emb.Encoder
    """Encoder model used to generate vector embeddings for sentences."""
    index: faiss.IndexFlatIP
    """Index for vector search."""
    sentences: list[str]
    """Input sentences."""
    batch_size: int
    """Size of the batch during database construction."""

    def __init__(
        self,
        encoder: emb.Encoder,
        index: faiss.IndexFlatIP,
        sentences: list[str],
        batch_size: int,
    ) -> None:
        """Use `VectorDatabase.load` or `VectorDatabase.empty` to construct a new DB."""
        self.encoder = encoder
        self.index = index
        self.sentences = sentences
        self.batch_size = batch_size

    @classmethod
    def empty(
        cls,
        encoder_model: str = DEFAULT_SENTENCE_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Self:
        """Create a new empty vector database with the given encoder.

        Args:
            encoder_model: Name of the model used to create sentence embeddings.
            batch_size: Number of sentences per batch when adding sentences to the index.
        """
        encoder = emb.Encoder(encoder_model)
        index = faiss.IndexFlatIP(encoder.dimensions)
        return cls(encoder=encoder, index=index, sentences=[], batch_size=batch_size)

    @classmethod
    def load(cls, db_dir: Path, batch_size: int | None = None) -> Self:
        """Load a vector database from disk.

        Expects the files created by `save`. If `batch_size` is given, it overrides the
        one from the metadata.

        Args:
            db_dir: Directory containing the database files.
            batch_size: Optional batch size to override the saved value.

        Returns:
            A loaded VectorDatabase instance.

        Raises:
            FileNotFoundError: If the index or metadata files don't exist.
            json.JSONDecodeError: If the metadata file contains invalid JSON.
        """
        index_path = db_dir / cls.INDEX_FILE
        metadata_path = db_dir / cls.METADATA_FILE

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_bytes())

        return cls(
            index=index,
            encoder=emb.Encoder(metadata["model_name"]),
            sentences=metadata["sentences"],
            batch_size=batch_size or int(metadata["batch_size"]),
        )

    def add_sentences(self, sentences: Iterable[str]) -> None:
        """Add sentences to index in batches.

        Args:
            sentences: Iterable of sentences to add to the database.
        """
        sentences_ = list(sentences)

        self.sentences.extend(sentences_)
        self.index.add(  # type: ignore
            self.encoder.batch_encode(
                sentences_, batch_size=self.batch_size, progress=True
            )
        )

    def search(
        self,
        query_sentences: Sequence[str],
        k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[SearchResult]:
        """Search sentences in database. Returns top `k` with similarity over `threshold`."""
        query_vectors = self.encoder.batch_encode(
            query_sentences, batch_size=self.batch_size
        )
        scores, indices = self.index.search(query_vectors, k)  # type: ignore

        return [
            SearchResult.model_construct(
                query=query,
                matches=[
                    SearchMatch.model_construct(
                        sentence=self.sentences[idx], score=float(score)
                    )
                    for score, idx in zip(scores_row, indices_row)
                    if score >= threshold
                ],
            )
            for query, scores_row, indices_row in zip(query_sentences, scores, indices)
        ]

    def find_best(
        self,
        query_sentences: Sequence[str],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[SearchMatch | None]:
        """Find the best match for each query sentence.

        Returns the highest similarity match for each query sentence, or None if no
        match is found above the threshold.

        Args:
            query_sentences: Sentences to search for in the database.
            threshold: Minimum similarity score to consider a match.

        Returns:
            A list of SearchMatch objects (one per query) or None where no match was
            found.
        """
        return [
            next(iter(result.matches), None)
            for result in self.search(query_sentences, threshold=threshold, k=1)
        ]

    def save(self, db_dir: Path) -> None:
        """Save database to `db_dir`.

        Saves two files:
        - `VectorDatabase.INDEX_FILE`
        - `VectorDatabase.METADATA_FILE`

        Args:
            db_dir: Directory where to save the database files.

        Raises:
            OSError: If there's an error creating the directory or writing the files.
        """
        db_dir.mkdir(parents=True, exist_ok=True)

        index_path = db_dir / self.INDEX_FILE
        metadata_path = db_dir / self.METADATA_FILE

        faiss.write_index(self.index, str(index_path))

        metadata_path.write_text(
            json.dumps({
                "batch_size": self.batch_size,
                "model_name": self.encoder.model_name,
                "sentences": self.sentences,
            })
        )
