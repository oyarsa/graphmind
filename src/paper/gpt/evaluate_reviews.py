"""Predict each review's novelty rating based on the review text.

The input is the processed PeerRead dataset (peerread.Paper).
"""

import asyncio
import logging
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI

from paper import peerread as pr
from paper.evaluation_metrics import Metrics, calculate_metrics
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    Demonstration,
    GPTFull,
    apply_rating_mode,
    fix_classified_rating,
)
from paper.gpt.model import (
    PaperWithReviewEval,
    Prompt,
    PromptResult,
    ReviewEvaluation,
)
from paper.gpt.prompts import PromptTemplate, load_prompts
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
    cli,
    display_params,
    ensure_envvar,
    progress,
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

REVIEW_CLASSIFY_USER_PROMPTS = load_prompts("evaluate_reviews")

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
    user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for classification.",
            click_type=cli.choice(REVIEW_CLASSIFY_USER_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results"),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed used for the GPT API.")] = 0,
    demos: Annotated[
        str | None,
        typer.Option(
            help="Name of file containing demonstrations to use in few-shot prompt",
            click_type=cli.choice(EVALUATE_DEMONSTRATIONS),
        ),
    ] = None,
    demo_prompt: Annotated[
        str,
        typer.Option(
            help="User prompt to use for building the few-shot demonstrations.",
            click_type=cli.choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ] = "abstract",
    mode: Annotated[
        str,
        typer.Option(
            help="Which mode to apply to target ratings.",
            click_type=cli.choice(range(-1, 6)),
        ),
    ] = "0",
) -> None:
    """Evaluate each review's novelty rating based on the review text."""
    asyncio.run(
        evaluate_reviews(
            model,
            peerread_path,
            limit_papers,
            user_prompt,
            output_dir,
            continue_papers,
            continue_,
            seed,
            demos,
            demo_prompt,
            int(mode),
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


def format_template(
    paper: pr.Paper,
    review: pr.PaperReview,
    user_prompt: PromptTemplate,
    demonstrations: str,
) -> str:
    """Format evaluation template using a peer review as reference."""
    return user_prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        review=review.rationale,
        demonstrations=demonstrations,
    )


async def evaluate_reviews(
    model: str,
    peerread_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    demonstrations_key: str | None,
    demo_prompt_key: str,
    mode: int,
) -> None:
    """Evaluate each review's novelty rating based on the review text.

    Args:
        model: GPT model code to use.
        peerread_path: Path to the JSON file containing the input papers data.
        limit_papers: Number of papers to process. If 0 or None, process all.
        user_prompt_key: Key to the user prompt to use for paper evaluation. See
            `REVIEW_CLASSIFY_USER_PROMPTS` for available options or `list_prompts` for
            more.
        output_dir: Directory to save the output files.
        continue_papers_file: If provided, check for entries in the input data.
        continue_: If True, use data from `continue_papers_file`.
        seed: Seed for the OpenAI API call.
        demonstrations_key: Name of demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            avaialble options or `list_prompts` for more.
        mode: Which mode to apply to ratings. See `apply_rating_mode`.
    """
    random.seed(seed)
    params = display_params()
    logger.info(params)

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    papers = shuffled(load_data(peerread_path, pr.Paper))[:limit_papers]
    logger.info("%s", _display_label_dist(papers, mode))

    user_prompt = REVIEW_CLASSIFY_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        EVALUATE_DEMONSTRATIONS[demonstrations_key] if demonstrations_key else []
    )
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]
    demonstrations = _format_demonstrations(
        demonstration_data, demonstration_prompt, mode
    )

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
            client,
            model,
            user_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            seed,
            mode,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)
    metrics = _calculate_review_metrics(results_items)

    logger.info("%d reviews evaluated.", sum(len(p.reviews) for p in results_items))
    logger.info("Overall metrics:\n%s", metrics)

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "metrics.json", metrics)
    (output_dir / "params.txt").write_text(params)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


def _display_label_dist(papers: Sequence[pr.Paper], mode: int) -> str:
    gold_dist = Counter(
        apply_rating_mode(r.rating, mode) for p in papers for r in p.reviews
    )
    total = sum(gold_dist.values())
    return f"Gold label distribution ({total}):\n" + "\n".join(
        f"- {label}: {count}" for label, count in sorted(gold_dist.items())
    )


def _format_demonstrations(
    demonstrations: Sequence[Demonstration], prompt: PromptTemplate, mode: int
) -> str:
    """Format all `demonstrations` according to `prompt` as a single string.

    If `demonstrations` is empty, returns the empty string.
    """
    if not demonstrations:
        return ""

    output_all = [
        "-Demonstrations-\n"
        "The following are examples of other paper evaluations with their novelty"
        " ratings and rationales:\n",
    ]

    output_all.extend(
        prompt.template.format(
            title=demo.title,
            abstract=demo.abstract,
            main_text=demo.text,
            rationale=demo.rationale,
            rating=apply_rating_mode(demo.rating, mode),
        )
        for demo in demonstrations
    )
    return f"\n{"-" * 50}\n".join(output_all)


async def _evaluate_reviews(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[pr.Paper],
    output_intermediate_file: Path,
    demonstrations: str,
    seed: int,
    mode: int,
) -> GPTResult[list[PromptResult[PaperWithReviewEval]]]:
    """Evaluate each review in each paper.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use.
        user_prompt: User prompt template to use for classification to be filled.
        papers: Papers from the PeerRead dataset to evaluate.
        output_intermediate_file: File to write new results after each task.
        demonstrations: Text of demonstrations for few-shot prompting
        seed: Seed for the OpenAI API call.
        mode: Which mode to apply to ratings. See `apply_rating_mode`.

    Returns:
        List of papers with evaluated reviews wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperWithReviewEval]] = []
    total_cost = 0

    tasks = [
        _evaluate_paper_reviews(
            client, model, paper, user_prompt, demonstrations, seed, mode
        )
        for paper in papers
    ]

    for task in progress.as_completed(tasks, desc="Processing papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(
            PaperWithReviewEval, output_intermediate_file, result.result
        )

    return GPTResult(results, total_cost)


async def _evaluate_paper_reviews(
    client: AsyncOpenAI,
    model: str,
    paper: pr.Paper,
    user_prompt: PromptTemplate,
    demonstrations: str,
    seed: int,
    mode: int,
) -> GPTResult[PromptResult[PaperWithReviewEval]]:
    """Evaluate all reviews for a single paper.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use.
        paper: Paper from the PeerRead dataset to evaluate.
        user_prompt: User prompt template to use for classification to be filled.
        demonstrations: Text of demonstrations for few-shot prompting
        seed: Seed for the OpenAI call.
        mode: Which mode to apply to ratings. See `apply_rating_mode`.

    Returns:
        Paper with evaluated reviews wrapped in a GPTResult.
    """
    total_cost = 0
    evaluated_reviews: list[ReviewEvaluation] = []
    user_prompts: list[str] = []
    main_review: ReviewEvaluation | None = None

    for review in paper.reviews:
        user_prompt_text = format_template(paper, review, user_prompt, demonstrations)
        result = await run_gpt(
            GPTFull, client, _REVIEW_SYSTEM_PROMPT, user_prompt_text, model, seed=seed
        )
        total_cost += result.cost

        evaluated = fix_classified_rating(result.result or GPTFull.error())

        new_review = ReviewEvaluation(
            rating=apply_rating_mode(review.rating, mode),
            confidence=review.confidence,
            rationale=review.rationale,
            predicted_rating=apply_rating_mode(evaluated.rating, mode),
            predicted_rationale=evaluated.rationale,
        )

        if new_review.rationale == paper.review.rationale:
            main_review = new_review

        evaluated_reviews.append(new_review)
        user_prompts.append(user_prompt_text)

    assert main_review, "Main review must be retrived."
    user_prompt_full = f"\n\n{"-"*80}\n\n".join(user_prompts)

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
            prompt=Prompt(system=_REVIEW_SYSTEM_PROMPT, user=user_prompt_full),
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
