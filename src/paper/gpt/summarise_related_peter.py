"""Summarise related papers from PETER for inclusion in main paper evaluation prompt.

The input is the output of `paper.peerread`, i.e. the output of `gpt.annotate_paper`
(papers with extracted scientific terms) and `gpt.classify_contexts` (citations
contexts classified by polarity) with the related papers queried through the PETER
graph.

The output is similar to the input, but the related papers have extra summarised
information that can be useful for evaluating papers.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from paper import peter
from paper.gpt.model import (
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
    get_remaining_items,
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

PETER_SUMMARISE_USER_PROMPTS = load_prompts("summarise_related_peter")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(
    short_help="Run related paper summarisation.", help=__doc__, no_args_is_help=True
)
def run(
    ann_graph_file: Annotated[
        Path,
        typer.Option(
            "--ann-graph",
            help="JSON file containing the annotated PeerRead papers with graph results.",
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
        typer.Option("--limit", "-n", help="The number of PeerRead papers to process."),
    ] = 10,
    positive_prompt: Annotated[
        str,
        typer.Option(
            help="The summarisation prompt to use for positively related papers.",
            click_type=cli.Choice(PETER_SUMMARISE_USER_PROMPTS),
        ),
    ] = "positive",
    negative_prompt: Annotated[
        str,
        typer.Option(
            help="The summarisation prompt to use for negatively related papers.",
            click_type=cli.Choice(PETER_SUMMARISE_USER_PROMPTS),
        ),
    ] = "negative",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results."),
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed used for data shuffling.")] = 0,
) -> None:
    """Summarise PETER-related papers."""
    asyncio.run(
        summarise_related(
            model,
            ann_graph_file,
            limit_papers,
            positive_prompt,
            negative_prompt,
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


async def summarise_related(
    model: str,
    ann_graph_file: Path,
    limit_papers: int | None,
    positive_prompt_key: str,
    negative_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
) -> None:
    """Summarise PETER-related papers.

    The papers should come from PETER query results (`peter.peerread`).

    Args:
        model: GPT model code. Must support Structured Outputs.
        ann_graph_file: Path to the JSON file containing the annotated papers with their
            graph data.
        limit_papers: Number of papers to process. If None, process all.
        positive_prompt_key: Prompt key to use to summarise positively related papers.
        negative_prompt_key: Prompt key to use to summarise negatively related papers.
        output_dir: Directory to save the output files: intermediate and final results.
        continue_papers_file: If provided, check for entries in the input data. If they
            exist, we use those results and skip processing papers in them.
        continue_: If True, use existing data from `continue_papers_file`.
        seed: Random seed used for input data shuffling and for the GPT call.

    Note: See `PETER_SUMMARISE_USER_PROMPTS` for summarisation prompt options.

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

    papers = shuffled(load_data(ann_graph_file, peter.PaperResult))[:limit_papers]

    prompt_pol = {
        peter.ContextPolarity.POSITIVE: PETER_SUMMARISE_USER_PROMPTS[
            positive_prompt_key
        ],
        peter.ContextPolarity.NEGATIVE: PETER_SUMMARISE_USER_PROMPTS[
            negative_prompt_key
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperWithRelatedSummary,
        output_intermediate_file,
        continue_papers_file,
        papers,
        continue_,
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
        results = await _summarise_papers(
            client,
            model,
            prompt_pol,
            papers_remaining.remaining,
            output_intermediate_file,
            seed=seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    assert len(results_all) == len(papers)
    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "result_items.json", results_items)
    save_data(output_dir / "params.json", params)


async def _summarise_papers(
    client: AsyncOpenAI,
    model: str,
    prompt_pol: Mapping[peter.ContextPolarity, PromptTemplate],
    ann_graphs: Iterable[peter.PaperResult],
    output_intermediate_file: Path,
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperWithRelatedSummary]]]:
    """Summarise information from PETER-related papers for use as paper evaluation input.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use (must support Structured Outputs).
        prompt_pol: Prompt templates for related papers by polarity.
        ann_graphs: Annotated PeerRead papers with their graph data.
        output_intermediate_file: File to write new results after each task is completed.
        demonstrations: Text of demonstrations for few-shot prompting.
        seed: Seed for the OpenAI API call.

    Returns:
        List of classified papers and their prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperWithRelatedSummary]] = []
    total_cost = 0

    tasks = [
        _summarise_paper(client, model, ann_graph, prompt_pol, seed=seed)
        for ann_graph in ann_graphs
    ]

    for task in progress.as_completed(tasks, desc="Summarising related papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(
            PaperWithRelatedSummary, output_intermediate_file, result.result
        )

    return GPTResult(results, total_cost)


_PETER_SUMMARISE_SYSTEM_PROMPT = """\
Given the following target paper and a related paper, determine what are the important \
points for comparison between the two. Focus on how the papers are similar and how they \
differ. This information will be used to support a novelty assessment of  the target \
paper.
"""


async def _summarise_paper(
    client: AsyncOpenAI,
    model: str,
    ann_result: peter.PaperResult,
    prompt_pol: Mapping[peter.ContextPolarity, PromptTemplate],
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperWithRelatedSummary]]:
    output: list[PromptResult[PaperRelatedSummarised]] = []
    total_cost = 0

    for related in ann_result.results.related:
        result = await _summarise_paper_related(
            client,
            model,
            ann_result.paper,
            related,
            user_prompt=prompt_pol[related.polarity],
            seed=seed,
        )
        total_cost += result.cost

        output.append(result.result)

    return GPTResult(
        PromptResult(
            prompt=Prompt(
                system=_PETER_SUMMARISE_SYSTEM_PROMPT,
                user=f"\n\n{"-"*80}\n\n".join(x.prompt.user for x in output),
            ),
            item=PaperWithRelatedSummary(
                paper=ann_result.paper, related=PromptResult.unwrap(output)
            ),
        ),
        cost=total_cost,
    )


class GPTRelatedSummary(BaseModel):
    """Result of summarising key aspects of related papers."""

    model_config = ConfigDict(frozen=True)

    summary: Annotated[
        str,
        Field(
            description="Related paper summary with its key points about the main paper."
        ),
    ]


async def _summarise_paper_related(
    client: AsyncOpenAI,
    model: str,
    paper: PeerReadAnnotated,
    related: peter.PaperRelated,
    user_prompt: PromptTemplate,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperRelatedSummarised]]:
    user_prompt_text = format_template(user_prompt, paper, related)

    result = await run_gpt(
        GPTRelatedSummary,
        client,
        _PETER_SUMMARISE_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )

    summary = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperRelatedSummarised.from_related(
                related, summary=summary.summary if summary is not None else "<error>"
            ),
            prompt=Prompt(system=_PETER_SUMMARISE_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


def format_template(
    prompt: PromptTemplate, main: PeerReadAnnotated, related: peter.PaperRelated
) -> str:
    """Format related paper summarisation template."""
    return prompt.template.format(
        title_main=main.title,
        abstract_main=main.abstract,
        title_related=related.title,
        abstract_related=related.abstract,
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts(
        "PETER RELATED PAPER SUMMARISATION", PETER_SUMMARISE_USER_PROMPTS, detail=detail
    )


if __name__ == "__main__":
    app()
