"""Vector database for retrieving similar sentences by cosine similarity.

Uses SentenceTransformers to build sentence embeddings and Annoy to create the index.
"""
# pyright: basic

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

import orjson
import typer
from annoy import AnnoyIndex

from paper import embedding as emb
from paper.embedding import DEFAULT_SENTENCE_MODEL
from paper.types import Immutable
from paper.util.serde import read_file_bytes, write_file_bytes

DEFAULT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.6
# Number of trees for Annoy index (trade-off: build time vs accuracy)
DEFAULT_N_TREES = 50

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


class SearchMatch(Immutable):
    """Match from searching a query in the database."""

    sentence: str
    """Sentence retrieved from the database."""
    score: float
    """Similarity score with the query."""


class SearchResult(Immutable):
    """Search result with all matches for a query."""

    query: str
    """Sentence used to query the database."""
    matches: list[SearchMatch]
    """Retrieved sentences with scores."""


class VectorDatabase:
    """Database for embedded sentences."""

    INDEX_FILE = "index.ann"
    """File with indexed documents with vector embeddings."""
    METADATA_FILE = "metadata.json.zst"
    """File with JSON metadata needed to construct the database."""

    encoder: emb.Encoder
    """Encoder model used to generate vector embeddings for sentences."""
    index: AnnoyIndex
    """Annoy index for vector search."""
    sentences: list[str]
    """Input sentences."""
    batch_size: int
    """Size of the batch during database construction."""
    n_trees: int
    """Number of trees for the Annoy index."""
    _is_built: bool
    """Whether the index has been built (Annoy requires explicit build step)."""

    def __init__(
        self,
        encoder: emb.Encoder,
        index: AnnoyIndex,
        sentences: list[str],
        batch_size: int,
        n_trees: int = DEFAULT_N_TREES,
        is_built: bool = False,
    ) -> None:
        """Use `VectorDatabase.load` or `VectorDatabase.empty` to construct a new DB."""
        self.encoder = encoder
        self.index = index
        self.sentences = sentences
        self.batch_size = batch_size
        self.n_trees = n_trees
        self._is_built = is_built

    @classmethod
    def empty(
        cls,
        encoder_model: str = DEFAULT_SENTENCE_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_trees: int = DEFAULT_N_TREES,
    ) -> Self:
        """Create a new empty vector database with the given encoder.

        Args:
            encoder_model: Name of the model used to create sentence embeddings.
            batch_size: Number of sentences per batch when adding sentences to the index.
            n_trees: Number of trees for Annoy index (more trees = better accuracy, slower build).
        """
        encoder = emb.Encoder(encoder_model)
        dimensions = encoder.dimensions
        if dimensions is None:
            raise ValueError(
                f"Encoder {encoder_model} does not provide embedding dimensions"
            )

        # Angular distance is equivalent to cosine similarity for normalised vectors
        index = AnnoyIndex(dimensions, "angular")
        return cls(
            encoder=encoder,
            index=index,
            sentences=[],
            batch_size=batch_size,
            n_trees=n_trees,
            is_built=False,
        )

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

        metadata = orjson.loads(read_file_bytes(metadata_path))

        encoder = emb.Encoder(metadata["model_name"])
        dimensions = encoder.dimensions

        if dimensions is None:
            raise ValueError(
                f"Encoder {metadata['model_name']} does not provide embedding dimensions"
            )

        index = AnnoyIndex(dimensions, "angular")
        index.load(str(index_path))

        return cls(
            index=index,
            encoder=encoder,
            sentences=metadata["sentences"],
            batch_size=batch_size or int(metadata["batch_size"]),
            n_trees=metadata.get("n_trees", DEFAULT_N_TREES),
            is_built=True,
        )

    def add_sentences(self, sentences: Iterable[str]) -> None:
        """Add sentences to index in batches.

        Args:
            sentences: Iterable of sentences to add to the database.

        Note:
            After adding all sentences, the index must be built before searching.
        """
        if self._is_built:
            raise RuntimeError(
                "Cannot add sentences to a built index. Annoy indexes are immutable"
                " once built."
            )

        sentences_ = list(sentences)
        start_idx = len(self.sentences)
        self.sentences.extend(sentences_)

        vectors = self.encoder.batch_encode(
            sentences_, batch_size=self.batch_size, progress=True
        )

        # Add each vector to the index
        for i, vector in enumerate(vectors):
            self.index.add_item(start_idx + i, vector)

    def build(self) -> None:
        """Build the Annoy index.

        Must be called after adding all sentences and before searching. If the index is
        already built, does nothing.
        """
        if not self._is_built:
            self.index.build(self.n_trees)
            self._is_built = True
        else:
            logger.debug(
                "Tried to call build() on already built Annoy index. Nothing to do."
            )

    def search(
        self,
        query_sentences: Sequence[str],
        k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[SearchResult]:
        """Search sentences in database. Returns top `k` with similarity over `threshold`.

        db.build() must be called before using this search.

        Raises:
            ValueError: if the index hasn't be built (use db.build).
        """
        if not self._is_built:
            raise ValueError("Cannot search an unbuilt Annoy index")

        query_vectors = self.encoder.batch_encode(
            query_sentences, batch_size=self.batch_size
        )

        results: list[SearchResult] = []
        for query, query_vector in zip(query_sentences, query_vectors):
            # Annoy returns (indices, distances) where distances are angular distances
            indices, distances = self.index.get_nns_by_vector(
                query_vector, k, include_distances=True
            )

            matches: list[SearchMatch] = []
            for idx, dist in zip(indices, distances):
                # Angular distance ranges from 0 to 2, where 0 means identical vectors
                # Convert angular distance to cosine similarity
                similarity = 1.0 - (dist / 2.0)
                if similarity >= threshold:
                    matches.append(
                        SearchMatch.model_construct(
                            sentence=self.sentences[idx], score=similarity
                        )
                    )

            results.append(SearchResult.model_construct(query=query, matches=matches))

        return results

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

        Builds the index, if the it hasn't been built yet.

        Args:
            db_dir: Directory where to save the database files.

        Raises:
            OSError: If there's an error creating the directory or writing the files.
        """
        if not self._is_built:
            self.build()

        db_dir.mkdir(parents=True, exist_ok=True)

        index_path = db_dir / self.INDEX_FILE
        metadata_path = db_dir / self.METADATA_FILE

        self.index.save(str(index_path))

        write_file_bytes(
            metadata_path,
            orjson.dumps({
                "batch_size": self.batch_size,
                "model_name": self.encoder.model_name,
                "sentences": self.sentences,
                "n_trees": self.n_trees,
            }),
        )
