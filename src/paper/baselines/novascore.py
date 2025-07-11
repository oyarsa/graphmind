"""Novascore method from Ai et al 2024.

This module implements the NovaSCORE method for evaluating the novelty of academic papers
based on their Atomic Content Units (ACUs).
"""
# pyright: basic

import gc
import itertools
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import computed_field
from tqdm import tqdm

from paper import gpt
from paper import semantic_scholar as s2
from paper.evaluation_metrics import calculate_paper_metrics, display_metrics
from paper.types import Immutable
from paper.util import get_params, render_params, sample, setup_logging
from paper.util.cli import die
from paper.util.serde import load_data, load_data_jsonl, save_data, save_data_jsonl
from paper.vector_db import (
    DEFAULT_SENTENCE_MODEL,
    SearchMatch,
    SearchResult,
    VectorDatabase,
)

DEFAULT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_SCORE_THRESHOLD = 0.3
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
        Path | None,
        typer.Option("--db", help="Directory with vector database files to update."),
    ] = None,
    model: Annotated[
        str, typer.Option(help="Name of the sentence transformer model")
    ] = DEFAULT_SENTENCE_MODEL,
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
    ] = 500_000,
    seed: Annotated[int, typer.Option(help="Random seed for sampling.")] = 0,
) -> None:
    """Build a vector database from sentences in the `acus` field of input JSON documents.

    The input documents should be the output of `paper gpt acus` with `s2.Paper` (e.g.
    `peerrelated.json` from `paper construct`).

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
    rng = random.Random(seed)
    papers = sample(papers, limit_papers, rng)

    if not papers:
        die("Input file is empty.")

    sentences: list[str] = []
    for paper in papers:
        sentences.extend(paper.acus)

    sentences = sample(sentences, limit_sentences, rng)

    db.add_sentences(sentences)

    db.save(output_dir)
    logger.info(
        f"Built database with {len(sentences)} sentences from {len(papers)} documents"
    )


class PaperResult(Immutable):
    """Result of querying paper ACUs in vector database.

    Stores the results of searching a paper's ACUs in the vector database, including the
    original paper and all search results for each ACU.
    """

    paper: gpt.PaperWithACUs[s2.PaperWithS2Refs]
    """The original paper with its ACUs."""
    results: list[SearchResult]
    """Search results for each ACU in the paper."""


def _query_papers(
    db: VectorDatabase,
    output_file: Path,
    papers: Sequence[gpt.PaperWithACUs[s2.PaperWithS2Refs]],
    threshold: float,
    top_k: int,
    batch_size: int,
) -> int:
    total_queries = 0

    # This can consume a lot of memory depending on the size of the database and the
    # input, so we manually engage the GC to reduce the memory usage.

    with tqdm(total=len(papers), desc="Querying papers") as pbar:
        for batch in itertools.batched(papers, batch_size):
            for paper in batch:
                sentences = paper.acus
                total_queries += len(sentences)

                if not sentences:
                    logger.warning(f"Document {paper.id} has no ACUs to query")
                    continue

                query_results = db.search(sentences, k=top_k, threshold=threshold)
                result = PaperResult.model_construct(paper=paper, results=query_results)
                save_data_jsonl(output_file, result)

                # Explicitly free memory from results to reduce memory usage
                del result
                del query_results
                del sentences

                pbar.update(1)

            # Reduce memory usage
            gc.collect()
            gc.collect()

    return total_queries


@app.command(no_args_is_help=True)
def query(
    db_dir: Annotated[
        Path, typer.Option("--db", help="Directory with vector database files.")
    ],
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input JSON file with documents to query."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Path to directory where results will be saved."
        ),
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
    batch_size: Annotated[
        int, typer.Option(help="Size of batches when processing papers.")
    ] = 100,
) -> None:
    """Query the vector database with sentences from the papers ACUs.

    The input documents should be the output of `paper gpt acus` with `s2.PaperWithS2Refs`
    (e.g. `peerread_with_s2_references.json` from `paper construct`).

    The output is a JSON file that has each input paper along with the retrieved ACUs.
    """
    params = get_params()
    logger.info(render_params(params))

    db = VectorDatabase.load(db_dir)

    if limit_papers == 0:
        limit_papers = None

    papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithACUs[s2.PaperWithS2Refs]])
    )
    rng = random.Random(0)
    papers = sample(papers, limit_papers, rng)

    if not papers:
        die("Input file is empty.")

    logger.info(f"Loaded {len(papers)} documents")

    # Use JSONL files so we don't have to keep the whole result in memory
    output_file = output_dir / "result.jsonl"
    output_file.unlink(missing_ok=True)

    total_queries = _query_papers(db, output_file, papers, threshold, top_k, batch_size)

    logger.info(f"Processed {len(papers)} documents with {total_queries} queries")
    save_data(output_dir / "params.json", params)


def _evaluate_paper(
    paper_result: PaperResult,
    *,
    threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Calculate the novelty score for a paper using the NovaSCORE method.

    Args:
        paper_result: Paper result with ACUs and their search results.
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
    paper = paper_result.paper

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

    # Calculate the total novelty score
    total_score = 0.0
    for acu, result in zip(paper.acus, paper_result.results, strict=True):
        match = _find_best_match(result, threshold)
        novelty = 1 if match is None else 0

        salient = acu in salient_acus

        weight = weight_salient if salient else weight_nonsalient
        total_score += novelty * weight

    # Normalize by the number of ACUs
    return total_score / len(paper.acus)


def _find_best_match(result: SearchResult, threshold: float) -> SearchMatch | None:
    """Return best match in `result` if it's above the `threshold`, or None.

    This can be used to answer the question: "does this sentence have at least one item
    with high similarity?".
    """
    if not result.matches:
        return None

    best_match = max(result.matches, key=lambda m: m.score)
    if best_match.score >= threshold:
        return best_match
    return None


class PaperEvaluated(Immutable):
    """Result of evaluating a paper's novelty using the NovaSCORE method.

    This class stores the original paper, its calculated novelty score, and the binary
    novelty label derived from thresholding the score.
    """

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


class EvaluationConfig(Immutable):
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
    paper_results: list[PaperResult], config: EvaluationConfig
) -> list[PaperEvaluated]:
    """Run the NovaSCORE evaluation on a list of papers using saved query results.

    Args:
        paper_results: List of paper results with search results for each ACU.
        config: Evaluation configuration parameters.

    Returns:
        List of evaluated papers with novelty scores and labels.
    """
    results: list[PaperEvaluated] = []

    for paper_result in tqdm(paper_results, desc="Evaluating papers"):
        paper = paper_result.paper
        if not paper.acus:
            logger.warning(f"Paper {paper.id} has no ACUs, skipping")
            continue

        score = _evaluate_paper(
            paper_result,
            threshold=config.sim_threshold,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        )

        results.append(
            PaperEvaluated(
                paper=paper,
                novascore=score,
                novalabel=int(score >= config.score_threshold),
            )
        )

    return results


@app.command(no_args_is_help=True)
def evaluate(
    results_file: Annotated[
        Path,
        typer.Option("--results", help="Input JSON file with saved query results."),
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
    save_results: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Whether to save the results. If False, saves on the metrics.",
        ),
    ] = True,
) -> None:
    """Calculate novelty score using saved query results.

    Input is the output of the `query` command, which contains papers with their ACUs and
    search results for each ACU.
    """
    params = get_params()
    logger.info(render_params(params))

    if limit_papers == 0:
        limit_papers = None

    logger.info(f"Loading query results from {results_file}")
    rng = random.Random(0)
    paper_results = sample(
        load_data_jsonl(results_file, PaperResult), limit_papers, rng
    )

    logger.info(f"Loaded results for {len(paper_results)} papers")

    config = EvaluationConfig(
        sim_threshold=sim_threshold,
        score_threshold=score_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    results = run_evaluation(paper_results, config)

    metrics = calculate_paper_metrics(results)
    logger.info("%s\n", display_metrics(metrics, results))

    logger.info(f"Saving results to {output_dir}")
    save_data(output_dir / "metrics.json", metrics)
    save_data(output_dir / "params.json", params)
    if save_results:
        save_data(output_dir / "result.json.zst", results)


@app.callback(help=__doc__)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


if __name__ == "__main__":
    app()
