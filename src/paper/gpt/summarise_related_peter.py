"""Summarise related papers from PETER to include in main paper evaluation prompt.

The input is the output of `paper.asap`, i.e. the output of `gpt.annotate_paper`
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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Self

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from paper import peter
from paper.gpt.model import ASAPAnnotated, Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPT_SEMAPHORE,
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
from paper.util.serde import Record, load_data, save_data

logger = logging.getLogger(__name__)

PETER_SUMMARISE_USER_PROMPTS = load_prompts("summarise_related_peter")

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
        typer.Option("--limit", "-n", help="The number of ASAP papers to process."),
    ] = 10,
    positive_prompt: Annotated[
        str,
        typer.Option(
            help="The summarisation prompt to use for positively related papers.",
            click_type=cli.choice(PETER_SUMMARISE_USER_PROMPTS),
        ),
    ] = "positive",
    negative_prompt: Annotated[
        str,
        typer.Option(
            help="The summarisation prompt to use for negatively related papers.",
            click_type=cli.choice(PETER_SUMMARISE_USER_PROMPTS),
        ),
    ] = "negative",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option(help="Start from scratch, ignoring existing intermediate results"),
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
            clean_run,
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
    clean_run: bool,
    seed: int,
) -> None:
    """Summarise PETER-related papers.

    The papers should come from PETER query results (`peter.asap`).

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
        clean_run: If True, ignore `continue_papers_file` and run everything from
            scratch.
        seed: Random seed used for input data shuffling.

    Note: See `PETER_SUMMARISE_USER_PROMPTS` for summarisation prompt options.

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

    papers = shuffled(load_data(ann_graph_file, peter.PaperResult))[:limit_papers]

    prompt_positive = PETER_SUMMARISE_USER_PROMPTS[positive_prompt_key]
    prompt_negative = PETER_SUMMARISE_USER_PROMPTS[negative_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperWithRelatedSummary,
        output_intermediate_file,
        continue_papers_file,
        papers,
        clean_run,
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
        results = await _summarise_papers(
            client,
            model,
            prompt_positive,
            prompt_negative,
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


async def _summarise_papers(
    client: AsyncOpenAI,
    model: str,
    prompt_positive: PromptTemplate,
    prompt_negative: PromptTemplate,
    ann_graphs: Iterable[peter.PaperResult],
    output_intermediate_file: Path,
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperWithRelatedSummary]]]:
    """Summarise information from PETER-related papers for use as paper evaluation input.

    Args:
        client: OpenAI client to use GPT.
        model: GPT model code to use (must support Structured Outputs).
        prompt_positive: Prompt template for positively related papers.
        prompt_negative: Prompt template for negatively related papers.
        ann_graphs: Annotated ASAP papers with their graph data.
        output_intermediate_file: File to write new results after each task is completed.
        demonstrations: Text of demonstrations for few-shot prompting.
        seed: Seed for the OpenAI API call.

    Returns:
        List of classified papers and their prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperWithRelatedSummary]] = []
    total_cost = 0

    tasks = [
        _summarise_paper(
            client, model, ann_graph, prompt_positive, prompt_negative, seed=seed
        )
        for ann_graph in ann_graphs
    ]

    for task in progress.as_completed(tasks, desc="Classifying papers"):
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
differ. This information will be used to support an approval or rejection decision of \
the target paper.
"""


class PaperWithRelatedSummary(Record):
    """ASAP paper with its related papers formatted as prompt input."""

    paper: ASAPAnnotated
    related: Sequence[PaperRelatedSummarised]

    @property
    def id(self) -> str:
        """Identify graph result as the underlying paper's ID."""
        return self.paper.id


async def _summarise_paper(
    client: AsyncOpenAI,
    model: str,
    ann_result: peter.PaperResult,
    prompt_positive: PromptTemplate,
    prompt_negative: PromptTemplate,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperWithRelatedSummary]]:
    output: list[PromptResult[PaperRelatedSummarised]] = []
    total_cost = 0

    for prompt, papers in [
        (prompt_positive, ann_result.results.positive),
        (prompt_negative, ann_result.results.negative),
    ]:
        for related in papers:
            result = await _summarise_paper_related(
                client, model, ann_result.paper, related, prompt, seed=seed
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
    paper: ASAPAnnotated,
    related: peter.PaperRelated,
    user_prompt: PromptTemplate,
    *,
    seed: int,
) -> GPTResult[PromptResult[PaperRelatedSummarised]]:
    user_prompt_text = format_template(user_prompt, paper, related)

    async with GPT_SEMAPHORE:
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


class PaperRelatedSummarised(peter.PaperRelated):
    """PETER-related paper with summary."""

    summary: str

    @classmethod
    def from_related(cls, related: peter.PaperRelated, summary: str) -> Self:
        """PETER-related paper with generated summary."""
        return cls.model_validate(related.model_dump() | {"summary": summary})


def format_template(
    prompt: PromptTemplate, main: ASAPAnnotated, related: peter.PaperRelated
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
