"""Evaluate a paper's novelty based on its full-body text.

The input is the processed PeerRead dataset (peerread.Paper).
"""

import asyncio
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from paper import semantic_scholar as s2
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    GPTFull,
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
    fix_evaluated_rating,
    format_demonstrations,
)
from paper.gpt.model import Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
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
    get_params,
    progress,
    render_params,
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data

logger = logging.getLogger(__name__)
FULL_CLASSIFY_USER_PROMPTS = load_prompts("evaluate_paper_full")

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
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 1,
    user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for classification.",
            click_type=cli.Choice(FULL_CLASSIFY_USER_PROMPTS),
        ),
    ] = "simple-abs",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results"),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed used for data shuffling.")] = 0,
    demos: Annotated[
        str | None,
        typer.Option(
            help="Name of file containing demonstrations to use in few-shot prompt",
            click_type=cli.Choice(EVALUATE_DEMONSTRATIONS),
        ),
    ] = None,
    demo_prompt: Annotated[
        str,
        typer.Option(
            help="User prompt to use for building the few-shot demonstrations.",
            click_type=cli.Choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ] = "abstract",
) -> None:
    """Evaluate a paper's novelty based on its full-body text."""
    asyncio.run(
        evaluate_papers(
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
        )
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


async def evaluate_papers(
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
) -> None:
    """Evaluate a paper's novelty based on its full-body text.

    The papers should come from the PeerRead dataset as processed by the
    paper.peerread module.

    Args:
        model: GPT model code. Must support Structured Outputs.
        peerread_path: Path to the JSON file containing the input papers data.
        limit_papers: Number of papers to process. Defaults to 1 example. If None,
            process all.
        graph_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_GRAPH_USER_PROMPTS` for available options or `list_prompts` for more.
        user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_CLASSIFY_USER_PROMPTS` for available options or `list_prompts` for more.
        display: If True, show each graph on screen. This suspends the process until
            the plot is closed.
        output_dir: Directory to save the output files: serialised graphs (GraphML),
            plot images (PNG) and classification results (JSON), if classification is
            enabled.
        classify: If True, classify the papers based on the generated graph.
        continue_papers_file: If provided, check for entries in the input data. If they
            are there, we use those results and skip processing them.
        continue_: If True, use data from `continue_papers`.
        seed: Random seed used for shuffling.
        demonstrations_key: Name of demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            avaialble options or `list_prompts` for more.

    Returns:
        None. The output is saved to `output_dir`.
    """
    params = get_params()
    logger.info(render_params(params))

    random.seed(seed)

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    papers = shuffled(load_data(peerread_path, s2.PaperWithS2Refs))[:limit_papers]
    user_prompt = FULL_CLASSIFY_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        EVALUATE_DEMONSTRATIONS[demonstrations_key] if demonstrations_key else []
    )
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperResult, output_intermediate_file, continue_papers_file, papers, continue_
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    if continue_:
        logger.info(
            "Skipping %d items from the `continue` file.", len(papers_remaining.done)
        )

    with Timer() as timer:
        results = await _classify_papers(
            client,
            model,
            user_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            seed=seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    metrics = calculate_paper_metrics(results_items)
    logger.info("%s\n", display_metrics(metrics, results_items))

    assert len(results_all) == len(papers)
    (output_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperResult]]).dump_json(results_all, indent=2)
    )
    (output_dir / "result_items.json").write_bytes(
        TypeAdapter(list[PaperResult]).dump_json(results_items, indent=2)
    )
    (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (output_dir / "params.json").write_text(json.dumps(params))


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[s2.PaperWithS2Refs],
    output_intermediate_file: Path,
    demonstrations: str,
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperResult]]]:
    """Classify Papers into approved/not approved using the paper main text.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt: User prompt template to use for classification to be filled
        papers: Papers from the PeerRead dataset to classify
        output_intermediate_file: File to write new results after each task is completed
        demonstrations: Text of demonstrations for few-shot prompting
        seed: Seed for the OpenAI API call

    Returns:
        List of classified papers wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperResult]] = []
    total_cost = 0

    tasks = [
        _classify_paper(client, model, paper, user_prompt, demonstrations, seed=seed)
        for paper in papers
    ]

    for task in progress.as_completed(tasks, desc="Classifying papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(PaperResult, output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


_FULL_CLASSIFY_SYSTEM_PROMPT = (
    "Give an approval or rejection to a paper submitted to a high-quality scientific"
    " conference."
)


def format_template(
    prompt: PromptTemplate, paper: s2.PaperWithS2Refs, demonstrations: str
) -> str:
    """Format full-text evaluation template `paper` and `demonstrations`."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.main_text,
        demonstrations=demonstrations,
    )


async def _classify_paper(
    client: AsyncOpenAI,
    model: str,
    paper: s2.PaperWithS2Refs,
    user_prompt: PromptTemplate,
    demonstrations: str,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperResult]]:
    user_prompt_text = format_template(user_prompt, paper, demonstrations)
    result = await run_gpt(
        GPTFull,
        client,
        _FULL_CLASSIFY_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )

    classified = fix_evaluated_rating(result.result or GPTFull.error())

    return GPTResult(
        result=PromptResult(
            item=PaperResult.from_s2peer(
                paper, classified.rating, classified.rationale
            ),
            prompt=Prompt(system=_FULL_CLASSIFY_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts("FULL PAPER EVALUATION", FULL_CLASSIFY_USER_PROMPTS, detail=detail)


if __name__ == "__main__":
    app()
