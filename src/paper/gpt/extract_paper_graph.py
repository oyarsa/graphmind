"""Extract a hierarchical graph representing a paper from its full contents.

The input is the output of `gpt.summarise_related_peter`. These are the PETER-queried
papers with the related papers summarised. This then converts the paper content to a
graph, converts the graph to text and uses it as input alongside the PETER results.

The output is the input annotated papers with their extracted graphs.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from tqdm import tqdm

from paper.gpt.extract_graph import ExtractedGraph
from paper.gpt.graph_types import get_graph_type
from paper.gpt.model import (
    Graph,
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
    get_params,
    progress,
    read_resource,
    render_params,
    sample,
    setup_logging,
)
from paper.util.cli import Choice, die
from paper.util.serde import load_data, save_data
from paper.util.typing import maybe

logger = logging.getLogger(__name__)

GRAPH_EXTRACT_USER_PROMPTS = load_prompts("extract_graph")
PRIMARY_AREAS: Sequence[str] = tomllib.loads(
    read_resource("gpt.prompts", "primary_areas.toml")
)["primary_areas"]

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
    graph_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for graph extraction.",
            click_type=Choice(GRAPH_EXTRACT_USER_PROMPTS),
        ),
    ] = "full",
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
    batch_size: Annotated[
        int, typer.Option(help="Number of requests per batch.")
    ] = 100,
) -> None:
    """Extract hierarchical graphs from paper contents."""
    asyncio.run(
        extract_graphs(
            model,
            paper_file,
            limit_papers,
            graph_prompt,
            output_dir,
            continue_papers,
            continue_,
            seed,
            batch_size,
        )
    )


@app.callback()
def main() -> None:
    """Set up logging for all subcommands."""
    setup_logging()


async def extract_graphs(
    model: str,
    paper_file: Path,
    limit_papers: int | None,
    graph_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    batch_size: int,
) -> None:
    """Extract hierarchical graphs from paper contents.

    The papers should come from `gpt.summarise_related_peter`.

    Args:
        model: GPT model code. Must support Structured Outputs.
        paper_file: Path to the JSON file containing the annotated papers with their
            graph data and summarised related papers.
        limit_papers: Number of papers to process. If None or 0, process all.
        graph_prompt_key: Key to the user prompt to use for graph extraction. See
            `GRAPH_EXTRACT_USER_PROMPTS` for available options or the `prompts` command
            for more information.
        output_dir: Directory to save the output files: intermediate and final results.
        continue_papers_file: If provided, check for entries in the input data. If they
            are there, we use those results and skip processing them.
        continue_: If True, ignore `continue_papers` and run everything from scratch.
        seed: Random seed used for shuffling and for the GPT call.
        batch_size: Number of items per request batch.

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

    papers = sample(
        PromptResult.unwrap(
            load_data(paper_file, PromptResult[PaperWithRelatedSummary])
        ),
        limit_papers,
    )

    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    if not graph_prompt.system:
        die(f"Graph prompt '{graph_prompt.name}' does not have a system prompt.")

    output_intermediate_file, papers_remaining = init_remaining_items(
        ExtractedGraph, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _extract_graphs(
            client,
            graph_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            batch_size,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)

    if len(results_all) != len(papers):
        logger.warning(
            "Some papers are missing from the output. Input: %d. Output: %d.",
            len(papers),
            len(results_all),
        )


async def _extract_graphs(
    client: LLMClient,
    graph_prompt: PromptTemplate,
    papers: Sequence[PaperWithRelatedSummary],
    output_intermediate_file: Path,
    batch_size: int,
) -> GPTResult[list[PromptResult[ExtractedGraph]]]:
    """Extract hierarchical graphs from paper contents.

    Args:
        client: OpenAI client to use GPT.
        graph_prompt: Prompt for extracting the graph from the paper.
        papers: Annotated PeerRead papers with their summarised graph data.
        output_intermediate_file: File to write new results after paper is evaluated.
        batch_size: Number of items per batch.

    Returns:
        List of papers with their extracted graphs and prompts wrapped in a GPTResult.
    """
    results: list[PromptResult[ExtractedGraph]] = []
    total_cost = 0

    with tqdm(
        total=len(papers), desc="Evaluating papers", position=0, leave=True
    ) as pbar_papers:
        for batch in itertools.batched(papers, batch_size):
            tasks = [_extract_graph(client, paper, graph_prompt) for paper in batch]

            for task in progress.as_completed(
                tasks, desc="Evaluating batch", position=1, leave=False
            ):
                result = await task
                total_cost += result.cost

                results.append(result.result)
                append_intermediate_result(output_intermediate_file, result.result)

            pbar_papers.update(len(batch))

    return GPTResult(results, total_cost)


async def _extract_graph(
    client: LLMClient, paper: PaperWithRelatedSummary, prompt: PromptTemplate
) -> GPTResult[PromptResult[ExtractedGraph]]:
    prompt_text = format_graph_template(prompt, paper.paper)

    result = await client.run(
        get_graph_type(prompt.type_name), prompt.system, prompt_text
    )
    graph = (
        maybe(result.result)
        .map(lambda g: g.to_graph(title=paper.title, abstract=paper.abstract))
        .unwrap_f(Graph.empty)
    )

    if graph.is_empty():
        logger.warning(f"Paper '{paper.title}': invalid Graph")

    return GPTResult(
        result=PromptResult(
            item=ExtractedGraph(
                paper=paper.paper.paper,
                graph=graph,
            ),
            prompt=Prompt(system=prompt.system, user=prompt_text),
        ),
        cost=result.cost,
    )


def format_graph_template(prompt: PromptTemplate, paper: PeerReadAnnotated) -> str:
    """Format graph extraction template using annotated paper."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.paper.main_text,
        primary_areas=", ".join(PRIMARY_AREAS),
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[bool, typer.Option(help="Show full prompt text.")] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    for title, prompts in [
        ("GRAPH EXTRACTION", GRAPH_EXTRACT_USER_PROMPTS),
    ]:
        print_prompts(title, prompts, detail=detail)


if __name__ == "__main__":
    app()
