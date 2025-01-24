"""Evaluate a paper's novelty based on main paper graph with PETER-queried papers.

The input is the output of `gpt.summarise_related_peter`. These are the PETER-queried
papers with the related papers summarised. This then converts the paper content to a
graph, and uses it as input alongside the PETER results.

The output is the input annotated papers with a predicted novelty rating.
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

from paper import peerread as pr
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    GPTFull,
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
    fix_evaluated_rating,
    get_demonstrations,
)
from paper.gpt.extract_graph import GPTGraph, GraphResult
from paper.gpt.model import (
    Graph,
    PaperRelatedSummarised,
    PaperWithRelatedSummary,
    PeerReadAnnotated,
    Prompt,
    PromptResult,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    append_intermediate_result,
    init_remaining_items,
    run_gpt,
)
from paper.util import (
    Timer,
    cli,
    ensure_envvar,
    get_params,
    progress,
    render_params,
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

GRAPH_EVAL_USER_PROMPTS = load_prompts("evaluate_graph")
GRAPH_EXTRACT_USER_PROMPTS = load_prompts("extract_graph")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


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
        str,
        typer.Option(
            "--model", "-m", help="The model to use for both extraction and evaluation."
        ),
    ] = "gpt-4o-mini",
    limit_papers: Annotated[
        int,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 10,
    eval_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for paper evaluation.",
            click_type=cli.Choice(GRAPH_EVAL_USER_PROMPTS),
        ),
    ] = "simple",
    graph_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for graph extraction.",
            click_type=cli.Choice(GRAPH_EXTRACT_USER_PROMPTS),
        ),
    ] = "strict",
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
) -> None:
    """Evaluate paper novelty with a paper graph and summarised PETER related papers."""
    asyncio.run(
        evaluate_papers(
            model,
            paper_file,
            limit_papers,
            eval_prompt,
            graph_prompt,
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
    paper_file: Path,
    limit_papers: int | None,
    eval_prompt_key: str,
    graph_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    demonstrations_key: str | None,
    demo_prompt_key: str,
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
        graph_prompt_key: Key to the user prompt to use for graph extraction. See
            `GRAPH_EXTRACT_USER_PROMPTS` for available optoins or the `prompts` command
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

    papers = shuffled(
        PromptResult.unwrap(
            load_data(paper_file, PromptResult[PaperWithRelatedSummary])
        )
    )[:limit_papers]

    eval_prompt = GRAPH_EVAL_USER_PROMPTS[eval_prompt_key]
    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    output_intermediate_file, papers_remaining = init_remaining_items(
        GraphResult, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _evaluate_papers(
            client,
            model,
            eval_prompt,
            graph_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result

    results_items = [r.paper for r in PromptResult.unwrap(results_all)]
    metrics = calculate_paper_metrics(results_items)
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
    client: AsyncOpenAI,
    model: str,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    paper: Sequence[PaperWithRelatedSummary],
    output_intermediate_file: Path,
    demonstrations: str,
    seed: int,
) -> GPTResult[list[PromptResult[GraphResult]]]:
    """Evaluate paper novelty using a paper graph and PETER-related papers.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use (must support Structured Outputs).
        eval_prompt: Prompt template for novelty evaluation.
        graph_prompt: Prompt template for graph extraction.
        paper: Annotated PeerRead papers with their summarised graph data.
        output_intermediate_file: File to write new results after paper is evaluated.
        demonstrations: Text of demonstrations for few-shot prompting.
        seed: Seed for the OpenAI API call.

    Returns:
        List of evaluated papers and their prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[GraphResult]] = []
    total_cost = 0

    tasks = [
        _evaluate_paper(
            client, model, paper, eval_prompt, graph_prompt, demonstrations, seed
        )
        for paper in paper
    ]

    for task in progress.as_completed(tasks, desc="Evaluating papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(GraphResult, output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


_GRAPH_EVAL_SYSTEM_PROMPT = """\
Given the following target paper and a selection of related papers separated by whether \
they're supporting or contrasting the main paper, give a novelty rating to a paper \
submitted to a high-quality scientific conference.
"""
_GRAPH_EXTRACT_SYSTEM_PROMPT = """\
Given the following scientific paper, extract the important entities from the text and \
the relationships between them. The goal is to build a faithful and concise representation \
of the paper that captures the most important elements necessary to evaluate its novelty.
"""


async def _evaluate_paper(
    client: AsyncOpenAI,
    model: str,
    paper: PaperWithRelatedSummary,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    demonstrations: str,
    seed: int,
) -> GPTResult[PromptResult[GraphResult]]:
    if "graph" in eval_prompt.name:
        graph_prompt_text = format_graph_template(
            graph_prompt, paper.paper, demonstrations
        )
        graph_result = await run_gpt(
            GPTGraph,
            client,
            _GRAPH_EXTRACT_SYSTEM_PROMPT,
            graph_prompt_text,
            model,
            seed=seed,
        )
        graph = (
            graph_result.result.to_graph(title=paper.title, abstract=paper.abstract)
            if graph_result.result
            else Graph.empty()
        )
    else:
        graph_prompt_text = "<no graph>"
        graph = Graph.empty()

    eval_prompt_text = format_eval_template(eval_prompt, paper, graph, demonstrations)
    eval_result = await run_gpt(
        GPTFull,
        client,
        _GRAPH_EVAL_SYSTEM_PROMPT,
        eval_prompt_text,
        model,
        seed=seed,
    )

    eval_paper = paper.paper.paper
    evaluated = fix_evaluated_rating(eval_result.result or GPTFull.error())

    sep = f"\n\n{"-" * 80}\n\n"
    combined_system_prompt = (
        f"{_GRAPH_EXTRACT_SYSTEM_PROMPT}{sep}{_GRAPH_EVAL_SYSTEM_PROMPT}"
    )
    combined_user_prompt = f"{graph_prompt_text}{sep}{eval_prompt_text}"

    return GPTResult(
        result=PromptResult(
            item=GraphResult(
                paper=PaperResult.from_s2peer(
                    eval_paper, evaluated.rating, evaluated.rationale
                ),
                graph=graph,
            ),
            prompt=Prompt(system=combined_system_prompt, user=combined_user_prompt),
        ),
        cost=eval_result.cost,
    )


def format_graph_template(
    prompt: PromptTemplate, paper: PeerReadAnnotated, demonstrations: str
) -> str:
    """Format graph extraction template using annotated paper."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.paper.main_text,
        demonstrations=demonstrations,
    )


def format_eval_template(
    prompt: PromptTemplate,
    paper: PaperWithRelatedSummary,
    graph: Graph,
    demonstrations: str,
) -> str:
    """Format evaluation template using the paper graph and PETER-queried related papers."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        demonstrations=demonstrations,
        positive=_format_related(
            p for p in paper.related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=_format_related(
            p for p in paper.related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(),
    )


def _format_related(related: Iterable[PaperRelatedSummarised]) -> str:
    return "\n\n".join(
        f"Title: {paper.title}\nSummary: {paper.summary}\n" for paper in related
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[bool, typer.Option(help="Show full prompt text.")] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    for title, prompts in [
        ("GRAPH EXTRACTION", GRAPH_EXTRACT_USER_PROMPTS),
        ("GRAPH PAPER EVALUATION", GRAPH_EVAL_USER_PROMPTS),
    ]:
        print_prompts(title, prompts, detail=detail)


if __name__ == "__main__":
    app()
