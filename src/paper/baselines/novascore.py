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
import typer
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from paper import embedding as emb
from paper import gpt
from paper.util import setup_logging
from paper.util.serde import load_data, save_data

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
    doc_id: int
    """Index of the document where the sentence came from."""


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
    sentence_map: list[tuple[str, int]]
    """Pairs of (sentence, document index)."""
    batch_size: int
    """Size of the batch during database construction."""

    def __init__(
        self,
        encoder: emb.Encoder,
        index: faiss.IndexFlatIP,
        sentence_map: list[tuple[str, int]],
        batch_size: int,
    ) -> None:
        """Use `VectorDatabase.load` or `VectorDatabase.empty` to construct a new DB."""
        self.encoder = encoder
        self.index = index
        self.sentence_map = sentence_map
        self.batch_size = batch_size

    @classmethod
    def empty(cls, encoder: emb.Encoder, batch_size: int = 1000) -> Self:
        """Create a new empty vector database with the given encoder."""
        index = faiss.IndexFlatIP(encoder.dimensions)
        return cls(encoder=encoder, index=index, sentence_map=[], batch_size=batch_size)

    @classmethod
    def load(cls, db_dir: Path) -> Self:
        """Load a vector database from disk.

        Expects the following files inside `db_dir`:
        - `VectorDatabase.INDEX_FILE`
        - `VectorDatabase.METADATA_FILE`
        """

        index_path = db_dir / cls.INDEX_FILE
        metadata_path = db_dir / cls.METADATA_FILE

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_bytes())

        return cls(
            index=index,
            encoder=emb.Encoder(metadata["model_name"]),
            sentence_map=metadata["sentence_map"],
            batch_size=metadata["batch_size"],
        )

    def add_sentences(self, sentences: Iterable[str], doc_id: int) -> None:
        """Add sentences to index."""
        for batch in itertools.batched(sentences, self.batch_size):
            for sentence in batch:
                self.sentence_map.append((sentence, doc_id))

            self.index.add(self.encoder.encode(batch))  # type: ignore

    def search(
        self, query_sentences: Iterable[str], k: int = 5, threshold: float = 0.7
    ) -> list[SearchResult]:
        """Search sentences in database. Returns top K with similarity over `threshold`.

        Excludes exact matches (same sentence) from results to avoid retrieving the
        original query sentence when searching within the dataset used to build the database.
        """
        results: list[SearchResult] = []

        for batch in itertools.batched(query_sentences, self.batch_size):
            query_vectors = self.encoder.encode(batch)

            scores, indices = self.index.search(query_vectors, k)  # type: ignore

            for q_idx, (sentence_scores, sentence_indices) in enumerate(
                zip(scores, indices)
            ):
                matches: list[SearchMatch] = []
                query_sentence = batch[q_idx]

                for score, idx in zip(sentence_scores, sentence_indices):
                    if score >= threshold and idx < len(self.sentence_map):
                        original_sentence, doc_id = self.sentence_map[idx]
                        # Skip exact matches to avoid retrieving the original sentence
                        if original_sentence != query_sentence:
                            matches.append(
                                SearchMatch(
                                    sentence=original_sentence,
                                    score=float(score),
                                    doc_id=doc_id,
                                )
                            )

                results.append(SearchResult(query=query_sentence, matches=matches))

        return results

    def save(self, db_dir: Path) -> None:
        """Save database to `db_dir`.

        Saves two files:
        - `VectorDatabase.INDEX_FILE`
        - `VectorDatabase.METADATA_FILE`
        """
        db_dir.mkdir(parents=True, exist_ok=True)

        index_path = db_dir / self.INDEX_FILE
        faiss.write_index(self.index, str(index_path))

        (db_dir / self.METADATA_FILE).write_text(
            json.dumps({
                "batch_size": self.batch_size,
                "model_name": self.encoder.model_name,
                "sentence_map": self.sentence_map,
            })
        )


@app.command(no_args_is_help=True)
def build(
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input JSON file with documents to index."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory where to save the vector database files.",
        ),
    ],
    model: Annotated[
        str, typer.Option(help="Name of the sentence transformer model")
    ] = "all-mpnet-base-v2",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing")] = 1000,
) -> None:
    """Build a vector database from sentences in the acus field of input JSON documents."""
    db = VectorDatabase.empty(emb.Encoder(model), batch_size)

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

    db.save(output_dir)
    logger.info(
        f"Built database with {total_sentences} sentences from {total_docs} documents"
    )


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


@app.command(no_args_is_help=True)
def query(
    db_dir: Annotated[
        Path, typer.Option("--db", help="Directory with vector database files.")
    ],
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input JSON file with query documents."),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSON file for results.")
    ],
    top_k: Annotated[int, typer.Option(help="Number of top matches to return.")] = 5,
    threshold: Annotated[float, typer.Option(help="Similarity threshold.")] = 0.7,
    paper_type: Annotated[
        PaperType, typer.Option(help="Type of paper for the input data.")
    ] = PaperType.S2,
) -> None:
    """Query the vector database with sentences from the papers ACUs."""
    db = VectorDatabase.load(db_dir)

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


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


if __name__ == "__main__":
    app()
