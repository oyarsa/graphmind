"""Evaluate a paper's approval based on annotated papers with SciMON-derived terms.

The input is the output of `scimon.query_asap`, i.e. the output of `gpt.annotate_paper`
(papers with extracted scientific terms) with the related terms extracted through the
SciMON graph created by `scimon.build`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI

from paper import scimon
from paper.gpt.evaluate_paper import (
    CLASSIFY_TYPES,
    EVALUATE_DEMONSTRATION_PROMPTS,
    Demonstration,
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
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
    progress,
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

SCIMON_CLASSIFY_USER_PROMPTS = load_prompts("evaluate_paper_scimon")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    ann_graph_file: Annotated[
        Path,
        typer.Option(
            "--ann-graph",
            help="JSON file containing the annotated ASAP papers with graph results.",
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
            click_type=cli.choice(SCIMON_CLASSIFY_USER_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results."),
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
    ] = "abstract",
) -> None:
    """Evaluate a paper's approval based on SciMON graph-extracted terms."""
    asyncio.run(
        evaluate_papers(
            model,
            ann_graph_file,
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
    ann_graph_file: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    demonstrations_file: Path | None,
    demo_prompt_key: str,
) -> None:
    """Evaluate a paper's approval based on SciMON graph-extracted terms.

    The papers should come from ASAP-annotated papers from `gpt.annotate_paper`.

    Args:
        model: GPT model code. Must support Structured Outputs.
        ann_graph_file: Path to the JSON file containing the annotated papers with their
            graph data.
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

    papers = shuffled(load_data(ann_graph_file, scimon.AnnotatedGraphResult))[
        :limit_papers
    ]

    user_prompt = SCIMON_CLASSIFY_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
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
    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "result_items.json", results_items)
    save_data(output_dir / "metrics.json", metrics)


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    ann_graphs: Sequence[scimon.AnnotatedGraphResult],
    output_intermediate_file: Path,
    demonstrations: str,
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperResult]]]:
    """Classify Papers into approved/not approved using the paper main text.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use (must support Structured Outputs).
        user_prompt: User prompt template to use for classification to be filled.
        ann_graphs: Annotated ASAP papers with their graph data.
        output_intermediate_file: File to write new results after each task is completed.
        demonstrations: Text of demonstrations for few-shot prompting.
        seed: Seed for the OpenAI API call.

    Returns:
        List of classified papers and their prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperResult]] = []
    total_cost = 0

    tasks = [
        _classify_paper(
            client, model, ann_graph, user_prompt, demonstrations, seed=seed
        )
        for ann_graph in ann_graphs
    ]

    for task in progress.as_completed(tasks, desc="Classifying papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(PaperResult, output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


_SCIMON_CLASSIFY_SYSTEM_PROMPT = """\
Given inspiration sentences from related papers, give an approval or rejection decision \
to a paper submitted to a high-quality scientific conference.
"""


async def _classify_paper(
    client: AsyncOpenAI,
    model: str,
    ann_result: scimon.AnnotatedGraphResult,
    user_prompt: PromptTemplate,
    demonstrations: str,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperResult]]:
    user_prompt_text = format_template(user_prompt, ann_result, demonstrations)

    result = await run_gpt(
        CLASSIFY_TYPES[user_prompt.type_name],
        client,
        _SCIMON_CLASSIFY_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )

    paper = ann_result.ann.paper
    classified = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                reviews=paper.reviews,
                authors=paper.authors,
                sections=paper.sections,
                approval=paper.approval,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
                rationale=classified.rationale if classified else "<error>",
            ),
            prompt=Prompt(system=_SCIMON_CLASSIFY_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


def format_template(
    prompt: PromptTemplate, ann_result: scimon.AnnotatedGraphResult, demonstrations: str
) -> str:
    """Format evaluation template using annotated terms, graphs and `demonstrations`."""
    terms = "\n".join(
        f"Related {desc} ({len(terms)}):\n{_bullets(terms)}\n"
        for desc, terms in [
            ("paper titles", ann_result.result.citations),
            ("backgrounds", ann_result.result.semantic),
        ]
    )

    return prompt.template.format(
        title=ann_result.ann.paper.title,
        abstract=ann_result.ann.paper.abstract,
        demonstrations=demonstrations,
        terms=terms,
    )


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts(
        "SCIMON PAPER EVALUATION", SCIMON_CLASSIFY_USER_PROMPTS, detail=detail
    )


if __name__ == "__main__":
    app()
