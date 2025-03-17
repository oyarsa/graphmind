"""Novascore method from Ai et al 2024."""
# pyright: basic

import itertools
import json
import logging
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import faiss  # type: ignore
import numpy as np
import numpy.typing as npt
import typer
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from paper import gpt
from paper.util.serde import load_data, save_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer()


class SentenceVectorizer:
    """Vector encoder for sentences."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_batch(self, sentences: Iterable[str]) -> npt.NDArray[np.float32]:
        """Encode list of sentences into embedding matrix."""
        tensor = self.model.encode(list(sentences), show_progress_bar=False)
        return np.array(tensor).astype("float32")


class SearchMatch(BaseModel):
    """Match from searching a query in the database."""

    model_config = ConfigDict(frozen=True)

    sentence: str
    score: float
    doc_id: int


class SearchResult(BaseModel):
    """Search result with all matches for a query."""

    model_config = ConfigDict(frozen=True)

    query: str
    matches: list[SearchMatch]


class VectorDatabase:
    """Database for embedded sentences."""

    _INDEX_FILE = "index.faiss"
    _METADATA_FILE = "metadata.json"

    def __init__(
        self,
        vectorizer: SentenceVectorizer,
        index: faiss.IndexFlatIP,
        sentence_map: list[tuple[str, int]],
    ) -> None:
        """Private constructor - use factory methods instead."""
        self.vectorizer = vectorizer
        self.index = index
        self.sentence_map = sentence_map

    @classmethod
    def empty(cls, vectorizer: SentenceVectorizer) -> "VectorDatabase":
        """Create a new empty vector database with the given vectorizer."""
        index = faiss.IndexFlatIP(vectorizer.dimension)
        return cls(vectorizer=vectorizer, index=index, sentence_map=[])

    @classmethod
    def load(cls, db_dir: Path) -> Self:
        """Load a vector database from disk."""

        index_path = db_dir / cls._INDEX_FILE
        metadata_path = db_dir / cls._METADATA_FILE

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_bytes())

        return cls(
            index=index,
            vectorizer=SentenceVectorizer(
                model_name=metadata["model_name"],
                batch_size=metadata["batch_size"],
            ),
            sentence_map=metadata["sentence_map"],
        )

    def add_sentences(self, sentences: Iterable[str], doc_id: int) -> None:
        """Add sentences to index."""
        for batch in itertools.batched(sentences, self.vectorizer.batch_size):
            for sentence in batch:
                self.sentence_map.append((sentence, doc_id))

            self.index.add(self.vectorizer.encode_batch(batch))  # type: ignore

    def search(
        self, query_sentences: Iterable[str], k: int = 5, threshold: float = 0.7
    ) -> list[SearchResult]:
        """Search sentences in database. Returns top K with similarity over `threshold`."""
        results: list[SearchResult] = []
        batch_size = self.vectorizer.batch_size

        for batch in itertools.batched(query_sentences, batch_size):
            query_vectors = self.vectorizer.encode_batch(batch)

            scores, indices = self.index.search(  # type: ignore
                np.array(query_vectors).astype("float32"), k
            )

            for q_idx, (sentence_scores, sentence_indices) in enumerate(
                zip(scores, indices)
            ):
                # Filter by threshold and create results
                matches: list[SearchMatch] = []

                for score, idx in zip(sentence_scores, sentence_indices):
                    if score >= threshold and idx < len(self.sentence_map):
                        original_sentence, doc_id = self.sentence_map[idx]
                        matches.append(
                            SearchMatch(
                                sentence=original_sentence,
                                score=float(score),
                                doc_id=doc_id,
                            )
                        )

                results.append(SearchResult(query=batch[q_idx], matches=matches))

        return results

    def save(self, db_dir: Path) -> None:
        """Save database index and metadata to `db_dir`."""
        db_dir.mkdir(parents=True, exist_ok=True)

        index_path = db_dir / self._INDEX_FILE
        faiss.write_index(self.index, str(index_path))

        (db_dir / self._METADATA_FILE).write_text(
            json.dumps({
                "batch_size": self.vectorizer.batch_size,
                "model_name": self.vectorizer.model_name,
                "sentence_map": self.sentence_map,
            })
        )


@app.command()
def build(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON file with documents to index")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Output file where to save the vector database")
    ],
    model: Annotated[
        str, typer.Option(help="Name of the sentence transformer model")
    ] = "all-MiniLM-L6-v2",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 1000,
) -> None:
    """Build a vector database from sentences in the acus field of input JSON documents."""
    logger.info(f"Building vector database from {input_file} using model {model}")

    db = VectorDatabase.empty(
        SentenceVectorizer(model_name=model, batch_size=batch_size)
    )

    input_data = load_data(input_file, gpt.PromptResult[gpt.S2PaperWithACUs])
    input_batches = list(itertools.batched(input_data, batch_size))

    total_docs = 0
    total_sentences = 0

    for doc_batch in tqdm(input_batches):
        for doc_idx, doc in enumerate(doc_batch):
            sentences = doc.item.acus
            doc_global_id = total_docs + doc_idx
            db.add_sentences(sentences, doc_global_id)
            total_sentences += len(sentences)

        total_docs += len(doc_batch)

    db.save(output_file)
    logger.info(f"Done: {total_sentences} sentences from {total_docs} documents")


type PaperWithACUs = gpt.S2PaperWithACUs | gpt.PeerPaperWithACUs


class PaperType(StrEnum):
    """Whether the paper came from the S2 API or PeerRead dataset."""

    S2 = "s2"
    PeerRead = "peerread"

    def get_type(self) -> type[PaperWithACUs]:
        """Returns concrete model type for the paper."""
        match self:
            case self.S2:
                return gpt.S2PaperWithACUs
            case self.PeerRead:
                return gpt.PeerPaperWithACUs


class PaperResult(BaseModel):
    """Result of querying paper ACUs in vector database."""

    model_config = ConfigDict(frozen=True)

    paper: PaperWithACUs
    results: list[SearchResult]


@app.command()
def query(
    db_file: Annotated[Path, typer.Argument(help="Vector database file")],
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON file with query documents")
    ],
    output_file: Annotated[Path, typer.Argument(help="Output JSON file for results")],
    top_k: Annotated[int, typer.Option(help="Number of top matches to return")] = 5,
    threshold: Annotated[float, typer.Option(help="Similarity threshold")] = 0.7,
    paper_type: Annotated[
        PaperType, typer.Option(help="Type of paper for the input data")
    ] = PaperType.S2,
) -> None:
    """Query the vector database with sentences from the acus field of query documents."""
    logger.info(f"Querying vector database {db_file} with documents from {input_file}")

    db = VectorDatabase.load(db_file)

    logger.info(f"Loading input documents from {input_file}")
    input_docs = load_data(input_file, gpt.PromptResult[paper_type.get_type()])
    logger.info(f"Loaded {len(input_docs)} documents")

    results: list[PaperResult] = []
    total_queries = 0

    for doc in tqdm(input_docs, desc="Processing documents"):
        sentences = doc.item.acus
        query_results = db.search(sentences, k=top_k, threshold=threshold)
        total_queries += len(sentences)

        results.append(PaperResult(paper=doc.item, results=query_results))

    logger.info(f"Processed {len(results)} documents with {total_queries} queries.")
    save_data(output_file, results)


if __name__ == "__main__":
    app.callback()(lambda: None)
    app()
