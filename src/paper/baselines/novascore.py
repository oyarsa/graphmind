"""Novascore method from Ai et al 2024.

This module implements the NovaSCORE method for evaluating the novelty of academic papers
based on their Atomic Content Units (ACUs).
"""
# pyright: basic

import logging
import random
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, computed_field
from tqdm import tqdm

from paper import gpt
from paper import semantic_scholar as s2
from paper.evaluation_metrics import calculate_paper_metrics, display_metrics
from paper.util import get_params, render_params, setup_logging
from paper.util.cli import die
from paper.util.serde import load_data, save_data
from paper.vector_db import SearchResult, VectorDatabase

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
    limit_papers: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="The number of papers to process. Use 0 for all."
        ),
    ] = 10,
    limit_sentences: Annotated[
        int | None,
        typer.Option(
            "--sentences",
            "-s",
            help="The number of sentences to use to build the index. Use 0 for all.",
        ),
    ] = 200_000,
) -> None:
    """Build a vector database from sentences in the acus field of input JSON documents.

    If `--db` is given, we load an existing database and add to it. If not, we create
    a new from scratch with `--model`.
    """
    if db_dir is not None:
        db = VectorDatabase.load(db_dir, batch_size)
    else:
        db = VectorDatabase.empty(model, batch_size)

    if limit_papers == 0:
        limit_papers = None
    if limit_sentences == 0:
        limit_sentences = None

    logger.info(f"Loading input data from {input_file}")
    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithACUs[s2.Paper]])
    )
    if limit_papers is not None:
        papers = random.sample(papers, limit_papers)

    if not papers:
        die("Input file is empty.")

    sentences: list[str] = []
    for paper in papers:
        sentences.extend(paper.acus)

    if limit_sentences is not None:
        sentences = random.sample(sentences, limit_sentences)

    db.add_sentences(sentences)

    db.save(output_dir)
    logger.info(
        f"Built database with {len(sentences)} sentences from {len(papers)} documents"
    )


class PaperResult(BaseModel):
    """Result of querying paper ACUs in vector database.

    Stores the results of searching a paper's ACUs in the vector database, including the
    original paper and all search results for each ACU.
    """

    model_config = ConfigDict(frozen=True)

    paper: gpt.PaperWithACUs
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
    limit_papers: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="The number of papers to process. Use 0 for all."
        ),
    ] = 10,
) -> None:
    """Query the vector database with sentences from the papers ACUs."""
    db = VectorDatabase.load(db_dir)

    if limit_papers == 0:
        limit_papers = None

    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithACUs])
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

        query_results = db.search(sentences, k=top_k, threshold=threshold)
        total_queries += len(sentences)

        results.append(PaperResult(paper=paper, results=query_results))

    logger.info(f"Processed {len(results)} documents with {total_queries} queries")
    logger.info(f"Saving results to {output_file}")
    save_data(output_file, results)


def _evaluate_paper(
    db: VectorDatabase,
    paper: gpt.PaperWithACUs[s2.PaperWithS2Refs],
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

    acu_novelty_matches = db.find_best(paper.acus, threshold=threshold)

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

    paper: gpt.PaperWithACUs[s2.PaperWithS2Refs]
    """Original PeerRead input paper with extracted ACUs."""
    novascore: float
    """Score derived from the NovaSCORE method (0.0 to 1.0)."""
    novalabel: int
    """Binary version of the score given a threshold (0 or 1)."""

    @computed_field
    @property
    def y_true(self) -> int:
        """Gold label for novelty (binary)."""
        return self.paper.paper.label

    @computed_field
    @property
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
    db: VectorDatabase,
    papers: list[gpt.PaperWithACUs[s2.PaperWithS2Refs]],
    config: EvaluationConfig,
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
        typer.Option(
            "--limit", "-n", help="The number of papers to process. Use 0 for all."
        ),
    ] = 10,
    sim_threshold: Annotated[
        float, typer.Option(help="Similarity threshold.", min=0.0, max=1.0)
    ] = DEFAULT_SIMILARITY_THRESHOLD,
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
    """Calculate novelty score from input papers using the NovaSCORE database.

    Input is the output of `gpt.extract_acu` with `PeerRead` papers.
    """
    params = get_params()
    logger.info(render_params(params))

    if limit_papers == 0:
        limit_papers = None

    logger.info(f"Loading vector database from {db_dir}")
    db = VectorDatabase.load(db_dir)

    logger.info(f"Loading papers from {input_file}")
    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithACUs[s2.PaperWithS2Refs]])
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
