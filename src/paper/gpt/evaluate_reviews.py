"""Predict each review's novelty rating based on the review text.

The input is the processed PeerRead dataset (peerread.Paper).
"""

import asyncio
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI

from paper import peerread as pr
from paper.evaluation_metrics import Metrics, calculate_metrics
from paper.gpt.evaluate_paper import GPTFull, fix_classified_rating
from paper.gpt.model import (
    PaperWithReviewEval,
    Prompt,
    PromptResult,
    ReviewEvaluation,
)
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    append_intermediate_result,
    get_remaining_items,
    run_gpt,
)
from paper.util import (
    Timer,
    display_params,
    ensure_envvar,
    progress,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    peerread_path: Annotated[
        Path,
        typer.Option(
            "--peerread",
            help="The path to the JSON file containing the PeerRead papers data.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="The path to the output directory where the files will be saved.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="The model to use for the extraction."),
    ] = "gpt-4o-mini",
    limit_papers: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="The number of papers to process. Use 0 for all papers.",
        ),
    ] = 1,
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results"),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed used for the GPT API.")] = 0,
) -> None:
    """Evaluate each review's novelty rating based on the review text."""
    asyncio.run(
        evaluate_reviews(
            model,
            peerread_path,
            limit_papers,
            output_dir,
            continue_papers,
            continue_,
            seed,
        )
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


_REVIEW_SYSTEM_PROMPT = (
    "You are an expert at evaluating scientific paper reviews. Given a paper's title,"
    " abstract, and a reviewer's comments, predict what novelty rating (1-5) the reviewer"
    " would have given, where 1 is least novel and 5 is most novel."
)


def _format_user_prompt(paper: pr.Paper, review: pr.PaperReview) -> str:
    return (
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Review: {review.rationale}\n\n"
        "Based on this review text, what novelty rating (1-5) would the reviewer have"
        " given? Explain your reasoning and then give the rating."
    )


async def evaluate_reviews(
    model: str,
    peerread_path: Path,
    limit_papers: int | None,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
) -> None:
    """Evaluate each review's novelty rating based on the review text.

    Args:
        model: GPT model code to use.
        peerread_path: Path to the JSON file containing the input papers data.
        limit_papers: Number of papers to process. If 0 or None, process all.
        output_dir: Directory to save the output files.
        continue_papers_file: If provided, check for entries in the input data.
        continue_: If True, use data from `continue_papers_file`.
        seed: Seed for the OpenAI API call.
    """
    logger.info(display_params())

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    papers = load_data(peerread_path, pr.Paper)[:limit_papers]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"

    papers_remaining = get_remaining_items(
        PaperWithReviewEval,
        output_intermediate_file,
        continue_papers_file,
        papers,
        continue_,
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all in the `continues` file. Exiting."
        )
        return

    if continue_:
        logger.info(
            "Skipping %d items from the `continues` file.", len(papers_remaining.done)
        )

    with Timer() as timer:
        results = await _evaluate_reviews(
            client, model, papers_remaining.remaining, output_intermediate_file, seed
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)
    metrics = _calculate_review_metrics(results_items)

    logger.info("Overall metrics:\n%s", metrics)

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "result_items.json", results_items)
    save_data(output_dir / "metrics.json", metrics)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


async def _evaluate_reviews(
    client: AsyncOpenAI,
    model: str,
    papers: Sequence[pr.Paper],
    output_intermediate_file: Path,
    seed: int,
) -> GPTResult[list[PromptResult[PaperWithReviewEval]]]:
    """Evaluate each review in each paper.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use.
        papers: Papers from the PeerRead dataset to evaluate.
        output_intermediate_file: File to write new results after each task.
        seed: Seed for the OpenAI API call.

    Returns:
        List of papers with evaluated reviews wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperWithReviewEval]] = []
    total_cost = 0

    tasks = [_evaluate_paper_reviews(client, model, paper, seed) for paper in papers]

    for task in progress.as_completed(tasks, desc="Processing papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(
            PaperWithReviewEval, output_intermediate_file, result.result
        )

    return GPTResult(results, total_cost)


async def _evaluate_paper_reviews(
    client: AsyncOpenAI, model: str, paper: pr.Paper, seed: int
) -> GPTResult[PromptResult[PaperWithReviewEval]]:
    """Evaluate all reviews for a single paper.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use.
        paper: Paper from the PeerRead dataset to evaluate.
        seed: Seed for the OpenAI call.

    Returns:
        Paper with evaluated reviews wrapped in a GPTResult.
    """
    total_cost = 0
    evaluated_reviews: list[ReviewEvaluation] = []
    user_prompts: list[str] = []
    main_review: ReviewEvaluation | None = None

    for review in paper.reviews:
        user_prompt = _format_user_prompt(paper, review)
        result = await run_gpt(
            GPTFull, client, _REVIEW_SYSTEM_PROMPT, user_prompt, model, seed=seed
        )
        total_cost += result.cost

        evaluated = fix_classified_rating(result.result or GPTFull.error())

        new_review = ReviewEvaluation(
            rating=review.rating,
            confidence=review.confidence,
            rationale=review.rationale,
            predicted_rating=evaluated.rating,
            predicted_rationale=evaluated.rationale,
        )

        if new_review.rationale == paper.review.rationale:
            main_review = new_review

        evaluated_reviews.append(new_review)
        user_prompts.append(user_prompt)

    assert main_review, "Main review must be retrived."
    user_prompt = f"\n\n{"-"*80}\n\n".join(user_prompts)

    return GPTResult(
        PromptResult(
            item=PaperWithReviewEval(
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                sections=paper.sections,
                approval=paper.approval,
                references=paper.references,
                conference=paper.conference,
                # New
                reviews=evaluated_reviews,
                review=main_review,
                rationale=main_review.rationale,
                rating=main_review.rating,
            ),
            prompt=Prompt(system=_REVIEW_SYSTEM_PROMPT, user=user_prompt),
        ),
        total_cost,
    )


def _calculate_review_metrics(results: Iterable[PaperWithReviewEval]) -> Metrics:
    """Calculate metrics across all reviews."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for paper in results:
        for review in paper.reviews:
            if review.predicted_rating is not None:
                y_true.append(review.rating)
                y_pred.append(review.predicted_rating)

    return calculate_metrics(y_true, y_pred)


if __name__ == "__main__":
    app()
