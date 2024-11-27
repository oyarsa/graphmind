"""Evaluate a paper's approval based on its full-body text."""

# Best configuration:
#     Command:
#     $ uv run gpt eval_full run output/asap_balanced_50.json tmp/eval-full -n 0 \
#         --clean-run --user-prompt simple-abs --demos output/demonstrations_10.json \
#         --demo-prompt abstract -m 4o
#
#     2024-11-04 20:03:37 | INFO | paper.gpt.evaluate_paper_full:151 | CONFIG:
#     - model: 4o
#     - api_key: None
#     - data_path: /Users/italo/dev/paper-hypergraph/output/asap_balanced_50.json (dc592a4f)
#     - limit_papers: 0
#     - user_prompt_key: simple-abs
#     - output_dir: /Users/italo/dev/paper-hypergraph/tmp/eval-full (directory)
#     - continue_papers_file: None
#     - clean_run: True
#     - seed: 0
#     - demonstrations_file: /Users/italo/dev/paper-hypergraph/output/demonstrations_10.json (55baa321)
#     - demo_prompt_key: abstract
#
# Output:
#     - P   : 0.6286
#     - R   : 0.8800
#     - F1  : 0.7333
#     - Acc : 0.6800
#
#     Gold (P/N): 25/25 (50.00%)
#     Pred (P/N): 35/15 (70.00%)

import asyncio
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from paper.gpt.evaluate_paper import (
    CLASSIFY_TYPES,
    EVALUATE_DEMONSTRATION_PROMPTS,
    Demonstration,
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
    format_demonstrations,
)
from paper.gpt.model import Paper, Prompt, PromptResult
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
    progress,
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
    data_path: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the papers data."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The path to the output directory where the files will be saved."
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
            click_type=cli.choice(FULL_CLASSIFY_USER_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option(help="Start from scratch, ignoring existing intermediate results"),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed used for data shuffling.")] = 0,
    demos: Annotated[
        Path | None,
        typer.Option(help="File containing demonstrations to use in few-shot prompt"),
    ] = None,
    demo_prompt: Annotated[
        str,
        typer.Option(
            help="User prompt to use for building the few-shot demonstrations.",
            click_type=cli.choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ] = "simple",
) -> None:
    asyncio.run(
        evaluate_papers(
            model,
            data_path,
            limit_papers,
            user_prompt,
            output_dir,
            continue_papers,
            clean_run,
            seed,
            demos,
            demo_prompt,
        )
    )


@app.callback()
def main() -> None:
    setup_logging()


async def evaluate_papers(
    model: str,
    data_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    clean_run: bool,
    seed: int,
    demonstrations_file: Path | None,
    demo_prompt_key: str,
) -> None:
    """Evaluate a paper's approval based on its full-body text.

    The papers should come from the ASAP-Review dataset as processed by the
    paper.asap module.

    Args:
        model: GPT model code. Must support Structured Outputs.
        data_path: Path to the JSON file containing the input papers data.
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
        clean_run: If True, ignore `continue_papers` and run everything from scratch.
        seed: Random seed used for shuffling.
        demonstrations_file: Path to demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            avaialble options or `list_prompts` for more.

    Returns:
        None. The output is saved to `output_dir`.
    """
    logger.info(display_params())

    random.seed(seed)

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    papers = shuffled(load_data(data_path, Paper))[:limit_papers]
    user_prompt = FULL_CLASSIFY_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        TypeAdapter(list[Demonstration]).validate_json(demonstrations_file.read_bytes())
        if demonstrations_file is not None
        else []
    )
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperResult, output_intermediate_file, continue_papers_file, papers, clean_run
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    if clean_run:
        logger.info("Clean run: ignoring `continue` file and using the whole data.")
    else:
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


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[Paper],
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
        papers: Papers from the ASAP-Review dataset to classify
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


def format_template(prompt: PromptTemplate, paper: Paper, demonstrations: str) -> str:
    """Format full-text evaluation template `paper` and `demonstrations`."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.main_text(),
        demonstrations=demonstrations,
    )


async def _classify_paper(
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    user_prompt: PromptTemplate,
    demonstrations: str,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperResult]]:
    user_prompt_text = format_template(user_prompt, paper, demonstrations)
    result = await run_gpt(
        CLASSIFY_TYPES[user_prompt.type_name],
        client,
        _FULL_CLASSIFY_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )
    classified = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                reviews=paper.reviews,
                sections=paper.sections,
                approval=paper.approval,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
                rationale=classified.rationale if classified else "<error>",
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
    print_prompts("FULL PAPER EVALUATION", FULL_CLASSIFY_USER_PROMPTS, detail=detail)


if __name__ == "__main__":
    app()
