"""Evaluate a paper's novelty based on LLMs grounded with web search.

The input is the output of `gpt.summarise_related_peter`. The output is the input
annotated papers with a predicted novelty rating.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Self

import dotenv
import typer
from tqdm import tqdm

from paper.evaluation_metrics import calculate_paper_metrics, display_metrics
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    GPTFull,
    PaperResult,
    fix_evaluated_rating,
    get_demonstrations,
)
from paper.gpt.model import (
    PaperWithRelatedSummary,
    PeerReadAnnotated,
    Prompt,
    PromptResult,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    LLMClient,
    append_intermediate_result,
    init_remaining_items,
)
from paper.util import (
    Timer,
    cli,
    get_params,
    progress,
    removeprefix_icase,
    render_params,
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

SEARCH_EVAL_USER_PROMPTS = load_prompts("evaluate_search")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


class SearchModel(str):
    """Search model name. Must be either Gemini or GPT with search."""

    def __new__(cls, model: str) -> Self:
        """Validate that the model is a valid search model."""
        if "gemini" not in model and "search" not in model:
            raise ValueError("Model must be either Gemini or GPT with search")
        return super().__new__(cls, model)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    paper_file: Annotated[
        Path,
        typer.Option(
            "--papers",
            help="JSON file containing the annotated PeerRead papers with summarised"
            " graph results.",
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
        SearchModel,
        typer.Option(
            "--model",
            "-m",
            help="The model to use for evaluation. Must support search.",
            parser=SearchModel,
        ),
    ] = SearchModel("gpt-4o-mini-search"),  # noqa: B008
    limit_papers: Annotated[
        int,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 10,
    eval_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for paper evaluation.",
            click_type=cli.Choice(SEARCH_EVAL_USER_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option(
            "--continue",
            help="Use existing intermediate results.",
        ),
    ] = False,
    seed: Annotated[
        int, typer.Option(help="Random seed used for data shuffling and OpenAI API.")
    ] = 0,
    demos: Annotated[
        str | None,
        typer.Option(
            help="Name of file containing demonstrations to use in few-shot prompt.",
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
    batch_size: Annotated[
        int, typer.Option(help="Number of requests per batch.")
    ] = 100,
) -> None:
    """Evaluate paper novelty with a paper graph and summarised PETER related papers."""
    asyncio.run(
        evaluate_papers(
            model,
            paper_file,
            limit_papers,
            eval_prompt,
            output_dir,
            continue_papers,
            continue_,
            seed,
            demos,
            demo_prompt,
            batch_size,
        )
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


async def evaluate_papers(
    model: str,
    paper_file: Path,
    limit_papers: int | None,
    eval_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    demonstrations_key: str | None,
    demo_prompt_key: str,
    batch_size: int,
) -> None:
    """Evaluate paper novelty with a paper graph and summarised PETER related papers.

    The papers should come from `gpt.summarise_related_peter`.

    Args:
        model: GPT model code. Must support Structured Outputs.
        paper_file: Path to the JSON file containing the annotated papers with their
            graph data and summarised related papers.
        limit_papers: Number of papers to process. If None, process all.
        eval_prompt_key: Key to the user prompt to use for paper evaluation. See
            `GRAPH_EVAL_USER_PROMPTS` for available options or the `prompts` command
            for more information.
        output_dir: Directory to save the output files: intermediate and final results,
            and classification metrics.
        continue_papers_file: If provided, check for entries in the input data. If they
            are there, we use those results and skip processing them.
        continue_: If True, ignore `continue_papers` and run everything from scratch.
        seed: Random seed used for shuffling and for the GPT call.
        demonstrations_key: Key to the demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            available options or `list_prompts` for more.
        batch_size: Number of items per batch.

    Returns:
        None. The output is saved to `output_dir`.
    """
    params = get_params()
    logger.info(render_params(params))

    random.seed(seed)

    dotenv.load_dotenv()

    if limit_papers == 0:
        limit_papers = None

    client = LLMClient.new(model=model, seed=seed)

    papers = shuffled(
        PromptResult.unwrap(
            load_data(paper_file, PromptResult[PaperWithRelatedSummary])
        )
    )[:limit_papers]

    eval_prompt = SEARCH_EVAL_USER_PROMPTS[eval_prompt_key]
    if not eval_prompt.system:
        raise ValueError(
            f"Eval prompt {eval_prompt.name!r} does not have a system prompt."
        )

    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    output_intermediate_file, papers_remaining = init_remaining_items(
        PaperResult, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _evaluate_papers(
            client,
            eval_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            batch_size,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result

    results_items = list(PromptResult.unwrap(results_all))
    metrics = calculate_paper_metrics(results_items, results.cost)
    logger.info("%s\n", display_metrics(metrics, results_items))

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "metrics.json", metrics)
    save_data(output_dir / "params.json", params)

    if len(results_all) != len(papers):
        logger.warning(
            "Some papers are missing from the output. Input: %d. Output: %d.",
            len(papers),
            len(results_all),
        )


async def _evaluate_papers(
    client: LLMClient,
    eval_prompt: PromptTemplate,
    papers: Sequence[PaperWithRelatedSummary],
    output_intermediate_file: Path,
    demonstrations: str,
    batch_size: int,
) -> GPTResult[list[PromptResult[PaperResult]]]:
    """Evaluate paper novelty using a paper graph and PETER-related papers.

    Args:
        client: OpenAI client to use GPT.
        eval_prompt: Prompt template for novelty evaluation.
        papers: Annotated PeerRead papers with their summarised graph data.
        output_intermediate_file: File to write new results after paper is evaluated.
        demonstrations: Text of demonstrations for few-shot prompting.
        batch_size: Number of items per batch.

    Returns:
        List of evaluated papers and their prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperResult]] = []
    total_cost = 0

    with tqdm(
        total=len(papers), desc="Evaluating papers", position=0, leave=True
    ) as pbar_papers:
        for batch in itertools.batched(papers, batch_size):
            tasks = [
                _evaluate_paper(client, paper, eval_prompt, demonstrations)
                for paper in batch
            ]

            for task in progress.as_completed(
                tasks, desc="Evaluating batch", position=1, leave=False
            ):
                result = await task
                total_cost += result.cost

                results.append(result.result)
                append_intermediate_result(output_intermediate_file, result.result)

            pbar_papers.update(len(batch))

    return GPTResult(results, total_cost)


async def _evaluate_paper(
    client: LLMClient,
    paper: PaperWithRelatedSummary,
    eval_prompt: PromptTemplate,
    demonstrations: str,
) -> GPTResult[PromptResult[PaperResult]]:
    prompt_text = format_template(eval_prompt, paper.paper, demonstrations)
    system_prompt = eval_prompt.system
    result_str = await client.plain(system_prompt, prompt_text, search_level="low")
    result = result_str.map(_parse_result)

    if not result.result or not result.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid GPTFull (evaluation result)")

    evaluated = fix_evaluated_rating(result.result or GPTFull.error())

    return GPTResult(
        result=PromptResult(
            item=PaperResult.from_s2peer(
                paper.paper.paper, evaluated.label, evaluated.rationale
            ),
            prompt=Prompt(system=system_prompt, user=prompt_text),
        ),
        cost=result.cost,
    )


def format_template(
    prompt: PromptTemplate, paper: PeerReadAnnotated, demonstrations: str
) -> str:
    """Format graph extraction template using annotated paper."""
    return prompt.template.format(
        demonstrations=demonstrations,
        title=paper.title,
        abstract=paper.abstract,
        approval=paper.paper.approval,
    )


def _parse_result(text: str | None) -> GPTFull:
    """Parse the output text to get the label.

    The output should have the following format:

        Rationale: <text>

        Label: <0 or 1>

    To make this extra-lenient, we take everything that isn't the Label line as part of
    the rationale (excluding the Rationale bit).

    If the text is invalid, returns `GPTFull.error()`.
    """
    if not text:
        return GPTFull.error()

    label = None
    rationale: list[str] = []

    for line in text.splitlines():
        line_fold = line.strip().casefold()

        if line_fold.startswith("rationale:"):
            rationale.append(removeprefix_icase(line, "rationale:").strip())

        if not line_fold.startswith("label:"):
            rationale.append(line)
            continue

        rest = line_fold.removeprefix("label:").strip()
        try:
            label = int(rest[0])
        except Exception:
            rationale.append(line)

    if label is None:
        return GPTFull.error()

    return GPTFull(label=label, rationale="\n".join(rationale))


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[bool, typer.Option(help="Show full prompt text.")] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    for title, prompts in [
        ("SEARCH PAPER EVALUATION", SEARCH_EVAL_USER_PROMPTS),
    ]:
        print_prompts(title, prompts, detail=detail)


if __name__ == "__main__":
    app()
