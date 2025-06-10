"""Evaluate a paper's novelty based on main paper graph with PETER-queried papers.

The input is the output of `gpt.summarise_related_peter`. These are the PETER-queried
papers with the related papers summarised. This then converts the paper content to a
graph, converts the graph to text and uses it as input alongside the PETER results.

How the graph is converted is controlled by `--linearisation`:
- 'topo': topologically sorts the entities and creates paragraphs in that order for each
  one
- 'fluent': use a fluent template-based method to convert entities to natural text.

The output is the input annotated papers with a predicted novelty rating.
"""

from __future__ import annotations

import asyncio
import logging
import random
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer

from paper import peerread as pr
from paper.evaluation_metrics import (
    TargetMode,
    calculate_paper_metrics,
    display_regular_negative_macro_metrics,
)
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    GPTFull,
    GPTStructured,
    GPTUncertain,
    PaperResult,
    fix_evaluated_rating,
    get_demonstrations,
)
from paper.gpt.extract_graph import GraphResult
from paper.gpt.graph_types import get_graph_type
from paper.gpt.model import (
    Graph,
    LinearisationMethod,
    PaperRelatedSummarised,
    PaperWithRelatedSummary,
    PeerReadAnnotated,
    Prompt,
    PromptResult,
    RelatedPaperSource,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    LLMClient,
    append_intermediate_result_async,
    gpt_sequence,
    init_remaining_items,
)
from paper.util import (
    Timer,
    batch_map_with_progress,
    cli,
    get_params,
    read_resource,
    render_params,
    sample,
    seqcat,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

GRAPH_EVAL_USER_PROMPTS = load_prompts("evaluate_graph")
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
    eval_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for paper evaluation.",
            click_type=cli.Choice(GRAPH_EVAL_USER_PROMPTS),
        ),
    ] = "sans",
    graph_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for graph extraction.",
            click_type=cli.Choice(GRAPH_EXTRACT_USER_PROMPTS),
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
    linearisation: Annotated[
        LinearisationMethod,
        typer.Option(
            help="How to convert the extracted graph into text for evaluation."
        ),
    ] = LinearisationMethod.TOPO,
    batch_size: Annotated[
        int, typer.Option(help="Number of requests per batch.")
    ] = 100,
    sources: Annotated[
        list[RelatedPaperSource],
        typer.Option(help="What sources to use for related papers."),
    ] = [RelatedPaperSource.CITATIONS, RelatedPaperSource.SEMANTIC],  # noqa: B006
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
            linearisation,
            batch_size,
            set(sources),
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
    linearisation_method: LinearisationMethod,
    batch_size: int,
    sources: set[RelatedPaperSource],
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
            `GRAPH_EXTRACT_USER_PROMPTS` for available options or the `prompts` command
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
        linearisation_method: How to convert the extract graph into text for evaluation.
        batch_size: Number of items per batch.
        sources: What kinds of related paper sources to use.

    Returns:
        None. The output is saved to `output_dir`.
    """
    params = get_params()
    logger.info(render_params(params))

    rng = random.Random(seed)

    dotenv.load_dotenv()

    client = LLMClient.new_env(model=model, seed=seed)

    papers = sample(
        PromptResult.unwrap(
            load_data(paper_file, PromptResult[PaperWithRelatedSummary])
        ),
        limit_papers,
        rng,
    )

    eval_prompt = GRAPH_EVAL_USER_PROMPTS[eval_prompt_key]
    if not eval_prompt.system:
        raise ValueError(
            f"Eval prompt {eval_prompt.name!r} does not have a system prompt."
        )

    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    if not graph_prompt.system:
        raise ValueError(
            f"Graph prompt {graph_prompt.name!r} does not have a system prompt."
        )

    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    output_intermediate_file, papers_remaining = init_remaining_items(
        GraphResult, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _evaluate_papers(
            client,
            eval_prompt,
            graph_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            linearisation_method,
            batch_size,
            sources,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = seqcat(papers_remaining.done, results.result)
    results_items = [r.paper for r in PromptResult.unwrap(results_all)]

    logger.info("Metrics\n%s", display_regular_negative_macro_metrics(results_items))

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)
    save_data(output_dir / "metrics.json", calculate_paper_metrics(results_items))

    if len(results_all) != len(papers):
        logger.warning(
            "Some papers are missing from the output. Input: %d. Output: %d.",
            len(papers),
            len(results_all),
        )


async def _evaluate_papers(
    client: LLMClient,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    papers: Sequence[PaperWithRelatedSummary],
    output_intermediate_file: Path,
    demonstrations: str,
    linearisation_method: LinearisationMethod,
    batch_size: int,
    sources: set[RelatedPaperSource],
) -> GPTResult[Sequence[PromptResult[GraphResult]]]:
    """Evaluate paper novelty using a paper graph and PETER-related papers.

    Args:
        client: LLM client to use GPT.
        eval_prompt: Prompt template for novelty evaluation.
        graph_prompt: Prompt template for graph extraction.
        papers: Annotated PeerRead papers with their summarised graph data.
        output_intermediate_file: File to write new results after paper is evaluated.
        demonstrations: Text of demonstrations for few-shot prompting.
        linearisation_method: How to transform the extract graph into text for
            evaluation.
        batch_size: Number of items per batch.
        sources: What kinds of related paper sources to use.

    Returns:
        List of evaluated papers and their prompts wrapped in a GPTResult.
    """

    async def evaluate(
        paper: PaperWithRelatedSummary,
    ) -> GPTResult[PromptResult[GraphResult]]:
        result = await _evaluate_paper(
            client,
            paper,
            eval_prompt,
            graph_prompt,
            demonstrations,
            linearisation_method,
            sources,
        )
        await append_intermediate_result_async(output_intermediate_file, result.result)
        return result

    return gpt_sequence(
        await batch_map_with_progress(evaluate, papers, batch_size, name="papers")
    )


async def _evaluate_paper(
    client: LLMClient,
    paper: PaperWithRelatedSummary,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    demonstrations: str,
    linearisation_method: LinearisationMethod,
    sources: set[RelatedPaperSource],
) -> GPTResult[PromptResult[GraphResult]]:
    if "uncertain" in eval_prompt.name:
        target_mode = TargetMode.UNCERTAIN
    else:
        target_mode = TargetMode.BIN

    if "graph" in eval_prompt.name:
        graph_prompt_text = format_graph_template(graph_prompt, paper.paper)
        graph_system_prompt = graph_prompt.system
        graph_result = await client.run(
            get_graph_type(graph_prompt.type_name),
            graph_system_prompt,
            graph_prompt_text,
        )
        graph = (
            graph_result.result.to_graph(title=paper.title, abstract=paper.abstract)
            if graph_result.result
            else Graph.empty()
        )
        graph_cost = graph_result.cost

        if graph.is_empty():
            logger.warning(f"Paper '{paper.title}': invalid Graph")
    else:
        graph_system_prompt = None
        graph_prompt_text = None
        graph_cost = 0
        graph = Graph.empty()

    eval_prompt_text = format_eval_template(
        eval_prompt, paper, graph, demonstrations, linearisation_method, sources
    )
    eval_system_prompt = eval_prompt.system

    # Determine evaluation type based on prompt type_name
    if eval_prompt.type_name == "GPTStructured":
        eval_type = GPTStructured
    elif target_mode is TargetMode.UNCERTAIN:
        eval_type = GPTUncertain
    else:
        eval_type = GPTFull

    eval_result = await client.run(eval_type, eval_system_prompt, eval_prompt_text)

    if not eval_result.result or not eval_result.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid evaluation result")

    if isinstance(eval_result.result, GPTStructured):
        structured = eval_result.result or GPTStructured.error()
        fixed_label = fix_evaluated_rating(structured, target_mode).label
        paper_result = PaperResult.from_s2peer(
            paper.paper.paper,
            fixed_label,
            structured.rationale,
            structured_evaluation=structured,
        )
    else:
        evaluated = fix_evaluated_rating(
            eval_result.result or GPTFull.error(), target_mode
        )
        paper_result = PaperResult.from_s2peer(
            paper.paper.paper, evaluated.label, evaluated.rationale
        )

    return GPTResult(
        result=PromptResult(
            item=GraphResult.from_annotated(
                paper=paper_result,
                graph=graph,
                related=paper.related,
                terms=paper.paper.terms,
                background=paper.paper.background,
                target=paper.paper.target,
            ),
            prompt=Prompt(system=eval_system_prompt, user=eval_prompt_text),
        ),
        cost=graph_cost + eval_result.cost,
    )


def format_graph_template(prompt: PromptTemplate, paper: PeerReadAnnotated) -> str:
    """Format graph extraction template using annotated paper."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.paper.main_text,
        primary_areas=", ".join(PRIMARY_AREAS),
    )


def format_eval_template(
    prompt: PromptTemplate,
    paper: PaperWithRelatedSummary,
    graph: Graph,
    demonstrations: str,
    method: LinearisationMethod,
    sources: set[RelatedPaperSource],
) -> str:
    """Format evaluation template using the paper graph and PETER-queried related papers."""
    related = [p for p in paper.related if p.source in sources]
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        demonstrations=demonstrations,
        positive=_format_related(
            p for p in related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=_format_related(
            p for p in related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(method),
        approval=paper.paper.paper.approval,
    )


def _format_related(related: Iterable[PaperRelatedSummarised]) -> str:
    """Build prompt from related papers titles and summaries."""
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
