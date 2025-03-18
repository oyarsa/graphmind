"""Novascore method from Ai et al 2024.

This module implements the NovaSCORE method for evaluating the novelty of academic papers
based on their Atomic Content Units (ACUs).
"""
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
from pydantic import BaseModel, ConfigDict, computed_field
from tqdm import tqdm

from paper import embedding as emb
from paper import gpt
from paper.evaluation_metrics import calculate_paper_metrics, display_metrics
from paper.gpt.model import PeerPaperWithACUs
from paper.util import get_params, render_params, setup_logging
from paper.util.cli import die
from paper.util.serde import load_data, save_data

DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_SALIENT_ALPHA = 0
DEFAULT_SALIENT_BETA = 0.5
DEFAULT_SALIENT_GAMMA = 1

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
    doc_id: str
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
    sentence_map: list[tuple[str, str]]
    """Pairs of (sentence, document index)."""
    batch_size: int
    """Size of the batch during database construction."""

    def __init__(
        self,
        encoder: emb.Encoder,
        index: faiss.IndexFlatIP,
        sentence_map: list[tuple[str, str]],
        batch_size: int,
    ) -> None:
        """Use `VectorDatabase.load` or `VectorDatabase.empty` to construct a new DB."""
        self.encoder = encoder
        self.index = index
        self.sentence_map = sentence_map
        self.batch_size = batch_size

    @classmethod
    def empty(cls, encoder: emb.Encoder, batch_size: int = DEFAULT_BATCH_SIZE) -> Self:
        """Create a new empty vector database with the given encoder.

        Args:
            encoder: Method used to convert input sentences to vectors.
            batch_size: Number of sentences per batch when adding sentences to the index.
        """
        index = faiss.IndexFlatIP(encoder.dimensions)
        return cls(encoder=encoder, index=index, sentence_map=[], batch_size=batch_size)

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
            sentence_map=metadata["sentence_map"],
            batch_size=batch_size or int(metadata["batch_size"]),
        )

    def add_sentences(self, sentences: Iterable[str], doc_id: str) -> None:
        """Add sentences to index in batches.

        `doc_id` represents the document where the sentences came from. If the sentences
        may come from different documents, call this function multiple times, one for
        each document.

        Args:
            sentences: Iterable of sentences to add to the database.
            doc_id: Identifier for the document containing these sentences.
        """
        # Convert to list to avoid consuming the iterable multiple times
        sentences_list = list(sentences)
        if not sentences_list:
            logger.warning(f"No sentences to add for document {doc_id}")
            return

        # Process in batches for efficiency
        for batch in itertools.batched(sentences_list, self.batch_size):
            # Prepare sentence map entries for this batch
            batch_entries = [(sentence, doc_id) for sentence in batch]
            self.sentence_map.extend(batch_entries)

            # Encode and add to the index
            self.index.add(self.encoder.encode(batch))  # type: ignore

    def search(
        self,
        query_sentences: Iterable[str],
        k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        query_doc_id: str | None = None,
        max_retrieval_factor: int = 5,
    ) -> list[SearchResult]:
        """Search sentences in database. Returns top `k` with similarity over `threshold`.

        Excludes matches from the same document (based on document ID) to avoid
        retrieving sentences from the same paper when searching within the dataset used
        to build the database.

        Uses batch processing for efficient vector search with a retrieval factor
        multiplier to ensure enough matches after filtering.
        """
        actual_k = k * max_retrieval_factor
        results: list[SearchResult] = []

        for sentences_batch in itertools.batched(query_sentences, self.batch_size):
            query_vectors = self.encoder.encode(sentences_batch)

            # Initial batch search with larger k to account for filtering
            scores_batch: list[list[float]]
            indices_batch: list[list[int]]
            scores_batch, indices_batch = self.index.search(query_vectors, actual_k)  # type: ignore

            for query_sentence, scores, indices in zip(
                sentences_batch, scores_batch, indices_batch
            ):
                matches: list[SearchMatch] = []

                for score, idx in zip(scores, indices):
                    # Skip if score is below threshold or index is invalid
                    if score < threshold or idx >= len(self.sentence_map):
                        continue

                    original_sentence, doc_id = self.sentence_map[idx]

                    # Skip matches from the same document
                    if doc_id == query_doc_id:
                        continue

                    matches.append(
                        SearchMatch(
                            sentence=original_sentence,
                            score=float(score),
                            doc_id=doc_id,
                        )
                    )

                if len(matches) < k:
                    logger.warning(
                        "Not enough matches for sentence after retrieving %d results."
                        " Query document ID: %s",
                        actual_k,
                        query_doc_id,
                    )

                # Sort by score and take top k
                matches.sort(key=lambda m: m.score, reverse=True)
                results.append(SearchResult(query=query_sentence, matches=matches[:k]))

        return results

    def find_best(
        self,
        query_sentences: Iterable[str],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        query_doc_id: str | None = None,
        max_retrieval_factor: int = 5,
    ) -> list[SearchMatch | None]:
        """Find the best match for each query sentence.

        Returns the highest similarity match for each query sentence, or None if no
        match is found above the threshold. Excludes matches from the same document.

        Args:
            query_sentences: Sentences to search for in the database.
            threshold: Minimum similarity score to consider a match.
            query_doc_id: ID of the document containing the query sentences (to exclude
                matches)>
            max_retrieval_factor: Multiplier for how many results to retrieve initially.

        Returns:
            A list of SearchMatch objects (one per query) or None where no match was
            found.
        """
        # We need to retrieve more than 1 result because we might filter some out later
        retrieval_k = max_retrieval_factor
        results: list[SearchMatch | None] = []

        for sentences_batch in itertools.batched(query_sentences, self.batch_size):
            query_vectors = self.encoder.encode(sentences_batch)

            scores_batch: list[list[float]]
            indices_batch: list[list[int]]
            scores_batch, indices_batch = self.index.search(query_vectors, retrieval_k)  # type: ignore

            for scores, indices in zip(scores_batch, indices_batch):
                best_match: SearchMatch | None = None
                best_score = 0.0

                for score, idx in zip(scores, indices):
                    # Skip if score is below threshold or index is invalid
                    if score < threshold or idx >= len(self.sentence_map):
                        continue

                    original_sentence, doc_id = self.sentence_map[idx]

                    # Skip matches from the same document
                    if doc_id == query_doc_id:
                        continue

                    # Keep track of the best match so far
                    if best_match is None or score > best_score:
                        best_match = SearchMatch(
                            sentence=original_sentence,
                            score=float(score),
                            doc_id=doc_id,
                        )
                        best_score = score

                results.append(best_match)

        return results

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
                "sentence_map": self.sentence_map,
            })
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
    db_dir: Annotated[
        Path | None, typer.Option("--db", help="Directory with vector database files.")
    ] = None,
    model: Annotated[
        str, typer.Option(help="Name of the sentence transformer model")
    ] = DEFAULT_MODEL,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for processing")
    ] = DEFAULT_BATCH_SIZE,
    paper_type: Annotated[
        PaperType, typer.Option(help="Type of paper for the input data.")
    ] = PaperType.S2,
    limit_papers: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 10,
) -> None:
    """Build a vector database from sentences in the acus field of input JSON documents.

    If `--db` is given, we load an existing database and add to it. If not, we create
    a new from scratch with `--model`.
    """
    if db_dir is not None:
        db = VectorDatabase.load(db_dir, batch_size)
    else:
        db = VectorDatabase.empty(emb.Encoder(model), batch_size)

    logger.info(f"Loading input data from {input_file}")
    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[paper_type.get_type()])
    )[:limit_papers]

    if not papers:
        die("Input file is empty.")

    total_sentences = 0

    for paper in tqdm(papers, desc="Processing papers"):
        db.add_sentences(paper.acus, paper.id)
        total_sentences += len(paper.acus)

    db.save(output_dir)
    logger.info(
        f"Built database with {total_sentences} sentences from {len(papers)} documents"
    )


class PaperResult(BaseModel):
    """Result of querying paper ACUs in vector database.

    Stores the results of searching a paper's ACUs in the vector database, including the
    original paper and all search results for each ACU.
    """

    model_config = ConfigDict(frozen=True)

    paper: PaperWithACUs
    """The original paper with its ACUs."""
    results: list[SearchResult]
    """Search results for each ACU in the paper."""


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
    top_k: Annotated[
        int, typer.Option(help="Number of top matches to return.", min=1)
    ] = DEFAULT_TOP_K,
    threshold: Annotated[
        float, typer.Option(help="Similarity threshold.", min=0.0, max=1.0)
    ] = DEFAULT_SIMILARITY_THRESHOLD,
    paper_type: Annotated[
        PaperType, typer.Option(help="Type of paper for the input data.")
    ] = PaperType.S2,
    limit_papers: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 10,
) -> None:
    """Query the vector database with sentences from the papers ACUs."""
    db = VectorDatabase.load(db_dir)

    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[paper_type.get_type()])
    )[:limit_papers]

    if not papers:
        die("Input file is empty.")

    logger.info(f"Loaded {len(papers)} documents")

    results: list[PaperResult] = []
    total_queries = 0

    for paper in tqdm(papers, desc="Querying papers"):
        sentences = paper.acus
        if not sentences:
            logger.warning(f"Document {paper.id} has no ACUs to query")
            continue

        query_results = db.search(
            sentences, k=top_k, threshold=threshold, query_doc_id=paper.id
        )
        total_queries += len(sentences)

        results.append(PaperResult(paper=paper, results=query_results))

    logger.info(f"Processed {len(results)} documents with {total_queries} queries")
    logger.info(f"Saving results to {output_file}")
    save_data(output_file, results)


def _evaluate_paper(
    db: VectorDatabase,
    paper: PaperWithACUs,
    *,
    threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Calculate the novelty score for a paper using the NovaSCORE method.

    Args:
        db: Vector database to search for similar ACUs.
        paper: Paper with ACUs to evaluate.
        threshold: Similarity threshold for considering an ACU as non-novel.
        alpha: Weight parameter for the non-salient ACU formula.
        beta: Offset parameter for the salience ratio in the formula.
        gamma: Base weight parameter for non-salient ACUs.

    Returns:
        A novelty score between 0.0 and 1.0, where higher values indicate more novelty.

    The formula for non-salient ACU weight is:
        weight_nonsalient = min(1, alpha * (salience_ratio - beta)^3 + gamma)

    Where salience_ratio is the proportion of salient ACUs in the paper (number of
    salient ACUs divided by the total number of ACUs).
    """
    if not paper.acus:
        logger.warning(f"Paper {paper.id} has no ACUs, returning zero novelty score")
        return 0.0

    salient_acus = set(paper.salient_acus)

    salience_ratio = len(salient_acus) / len(paper.acus)
    weight_salient = 1.0  # Salient ACUs always have full weight

    # Calculate weight for non-salient ACUs using the formula
    weight_nonsalient = min(
        weight_salient,
        alpha * (salience_ratio - beta) ** 3 + gamma,
    )

    acu_novelty_matches = db.find_best(
        paper.acus, threshold=threshold, query_doc_id=paper.id
    )

    # Calculate the total novelty score
    total_score = 0.0
    for acu, match in zip(paper.acus, acu_novelty_matches):
        novelty = 1 if match is None else 0
        salient = acu in salient_acus

        weight = weight_salient if salient else weight_nonsalient
        total_score += novelty * weight

    # Normalize by the number of ACUs
    return total_score / len(paper.acus)


class PaperEvaluated(BaseModel):
    """Result of evaluating a paper's novelty using the NovaSCORE method.

    This class stores the original paper, its calculated novelty score, and the binary
    novelty label derived from thresholding the score.
    """

    model_config = ConfigDict(frozen=True)

    paper: PeerPaperWithACUs
    """Original PeerRead input paper with extracted ACUs."""
    novascore: float
    """Score derived from the NovaSCORE method (0.0 to 1.0)."""
    novalabel: int
    """Binary version of the score given a threshold (0 or 1)."""

    @property
    @computed_field
    def y_true(self) -> int:
        """Gold label for novelty (binary)."""
        return self.paper.paper.label

    @property
    @computed_field
    def y_pred(self) -> int:
        """Predicted label for novelty (binary)."""
        return self.novalabel


class EvaluationConfig(BaseModel):
    """Configuration for paper evaluation with NovaSCORE."""

    sim_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    """Threshold for similarity when determining novelty."""
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    """Threshold for converting NovaSCORE to binary label."""
    alpha: float
    """Weight parameter for non-salient ACU formula."""
    beta: float
    """Offset parameter for salience ratio in formula."""
    gamma: float
    """Base weight parameter for non-salient ACUs."""


def run_evaluation(
    db: VectorDatabase, papers: list[PeerPaperWithACUs], config: EvaluationConfig
) -> list[PaperEvaluated]:
    """Run the NovaSCORE evaluation on a list of papers.

    Args:
        db: Vector database to use for similarity search.
        papers: List of papers to evaluate.
        config: Evaluation configuration parameters.

    Returns:
        List of evaluated papers with novelty scores and labels.
    """
    results: list[PaperEvaluated] = []

    for paper in tqdm(papers, desc="Evaluating papers"):
        if not paper.acus:
            logger.warning(f"Paper {paper.id} has no ACUs, skipping")
            continue

        score = _evaluate_paper(
            db,
            paper,
            threshold=config.sim_threshold,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        )

        results.append(
            PaperEvaluated(
                paper=paper,
                novascore=score,
                novalabel=int(score > config.score_threshold),
            )
        )

    return results


@app.command(no_args_is_help=True)
def evaluate(
    db_dir: Annotated[
        Path, typer.Option("--db", help="Directory with vector database files.")
    ],
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input JSON file with query documents."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory to save the results."),
    ],
    limit_papers: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 10,
    sim_threshold: Annotated[
        float, typer.Option(help="Similarity threshold.", min=0.0, max=1.0)
    ] = DEFAULT_SIMILARITY_THRESHOLD,
    # TODO: figure out weights and threshold
    score_threshold: Annotated[
        float,
        typer.Option(help="Threshold for binary novelty.", min=0.0, max=1.0),
    ] = DEFAULT_SCORE_THRESHOLD,
    alpha: Annotated[
        float,
        typer.Option(help="Alpha parameter for dynamic salient weight.", min=0, max=2),
    ] = DEFAULT_SALIENT_ALPHA,
    beta: Annotated[
        float,
        typer.Option(help="Beta parameter for dynamic salient weight.", min=0, max=1),
    ] = DEFAULT_SALIENT_BETA,
    gamma: Annotated[
        float,
        typer.Option(help="Gamma parameter for dynamic salient weight.", min=0, max=1),
    ] = DEFAULT_SALIENT_GAMMA,
) -> None:
    """Query the vector database with sentences from the papers ACUs."""
    params = get_params()
    logger.info(render_params(params))

    if limit_papers == 0:
        limit_papers = None

    logger.info(f"Loading vector database from {db_dir}")
    db = VectorDatabase.load(db_dir)

    logger.info(f"Loading papers from {input_file}")
    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[PeerPaperWithACUs])
    )[:limit_papers]

    if not papers:
        raise typer.Exit(code=0)

    logger.info(f"Loaded {len(papers)} papers")

    config = EvaluationConfig(
        sim_threshold=sim_threshold,
        score_threshold=score_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    results = run_evaluation(db, papers, config)

    if not results:
        logger.warning("No results produced from evaluation")
        return

    metrics = calculate_paper_metrics(results, 0)
    logger.info("%s\n", display_metrics(metrics, results))

    logger.info(f"Saving results to {output_dir}")
    save_data(output_dir / "result.json", results)
    save_data(output_dir / "metrics.json", metrics)
    save_data(output_dir / "params.json", params)


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


if __name__ == "__main__":
    app()
