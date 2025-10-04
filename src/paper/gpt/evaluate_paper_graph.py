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

import typer

from paper import peerread as pr
from paper.evaluation_metrics import (
    TargetMode,
    calculate_paper_metrics,
    display_regular_negative_macro_metrics,
)
from paper.gpt.ensemble import GPTPreConfidence, aggregate_ensemble_evaluations
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    GPTFull,
    GPTStructured,
    GPTStructuredRaw,
    GPTUncertain,
    PaperResult,
    fix_evaluated_rating,
    get_demonstrations,
)
from paper.gpt.extract_graph import GraphResult
from paper.gpt.graph_types import get_graph_type
from paper.gpt.model import (
    RATIONALE_ERROR,
    Graph,
    LinearisationMethod,
    PaperRelatedSummarised,
    PaperWithRelatedSummary,
    PeerReadAnnotated,
    PromptResult,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    LLMClient,
    append_intermediate_result_async,
    gpt_sequence,
    gpt_unit,
    init_remaining_items,
)
from paper.types import PaperSource
from paper.util import (
    Timer,
    batch_map_with_progress,
    cli,
    dotenv,
    get_params,
    read_resource,
    render_params,
    sample,
    seqcat,
    setup_logging,
)
from paper.util.serde import Compress, load_data, save_data

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
    ] = "full-graph-structured",
    graph_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for graph extraction.",
            click_type=cli.Choice(GRAPH_EXTRACT_USER_PROMPTS),
        ),
    ] = "excerpts",
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
    temperature: Annotated[
        float,
        typer.Option(
            help="Temperature for the GPT model. 0 is deterministic, 1 is more random.",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0,
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
    n_evaluations: Annotated[
        int,
        typer.Option(
            help="Number of evaluation rounds per paper for ensemble voting.",
            min=1,
        ),
    ] = 10,
    eval_temperature: Annotated[
        float,
        typer.Option(
            help="Temperature for evaluation rounds. Higher values increase randomness.",
            min=0.0,
            max=2.0,
        ),
    ] = 0.8,
    batch_size: Annotated[
        int, typer.Option(help="Number of requests per batch.")
    ] = 100,
    sources: Annotated[
        list[PaperSource],
        typer.Option(help="What sources to use for related papers."),
    ] = [PaperSource.CITATIONS, PaperSource.SEMANTIC],  # noqa: B006
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
            temperature,
            demos,
            demo_prompt,
            linearisation,
            n_evaluations,
            eval_temperature,
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
    temperature: float,
    demonstrations_key: str | None,
    demo_prompt_key: str,
    linearisation_method: LinearisationMethod,
    n_evaluations: int,
    eval_temperature: float,
    batch_size: int,
    sources: set[PaperSource],
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
        temperature: Temperature for the GPT model.
        demonstrations_key: Key to the demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            available options or `list_prompts` for more.
        linearisation_method: How to convert the extract graph into text for evaluation.
        n_evaluations: Number of evaluation rounds per paper for ensemble voting.
        eval_temperature: Temperature for evaluation rounds. Higher values increase
            randomness in the LLM responses.
        batch_size: Number of items per batch.
        sources: What kinds of related paper sources to use.

    Returns:
        None. The output is saved to `output_dir`.
    """
    params = get_params()
    logger.info(render_params(params))

    rng = random.Random(seed)

    dotenv.load_dotenv()

    client = LLMClient.new_env(model=model, seed=seed, temperature=temperature)

    if "gemini" in model:
        # Maximum batch size for Geminmi is 25 because there's a hard limit on the number
        # of concurrent requests. That limit is 50, but we're playing it safe here.
        batch_size = min(batch_size, 25)

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
            n_evaluations,
            eval_temperature,
            batch_size,
            sources,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = seqcat(papers_remaining.done, results.result)
    results_items = [r.paper for r in PromptResult.unwrap(results_all)]

    logger.info("Metrics\n%s", display_regular_negative_macro_metrics(results_items))
    logger.info("Calls made: %d", client.calls_made)
    logger.info("Tokens used: %d", client.tokens_used)

    save_data(output_dir / "result.json.zst", results_all)
    save_data(output_dir / "params.json", params)
    save_data(
        output_dir / "metrics.json",
        calculate_paper_metrics(results_items, cost=results.cost),
        compress=Compress.NONE,
    )

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
    n_evaluations: int,
    eval_temperature: float,
    batch_size: int,
    sources: set[PaperSource],
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
        n_evaluations: Number of evaluation rounds per paper for ensemble voting.
        eval_temperature: Temperature for evaluation rounds.
        batch_size: Number of items per batch.
        sources: What kinds of related paper sources to use.

    Returns:
        List of evaluated papers and their prompts wrapped in a GPTResult.
    """

    async def evaluate(
        paper: PaperWithRelatedSummary,
    ) -> GPTResult[PromptResult[GraphResult]]:
        result = await evaluate_paper(
            client,
            paper,
            eval_prompt,
            graph_prompt,
            demonstrations,
            linearisation_method,
            n_evaluations,
            eval_temperature,
            sources,
        )
        await append_intermediate_result_async(output_intermediate_file, result.result)
        return result

    return gpt_sequence(
        await batch_map_with_progress(evaluate, papers, batch_size, name="papers")
    )


async def _run_evaluation_rounds(
    client: LLMClient,
    eval_type: type[GPTUncertain | GPTStructuredRaw | GPTFull],
    system_prompt: str,
    user_prompt: str,
    n_evaluations: int,
    eval_temperature: float,
) -> GPTResult[list[GPTUncertain | GPTStructuredRaw | GPTFull]]:
    """Run N evaluation rounds and return valid evaluations.

    Args:
        client: LLM client for API calls.
        eval_type: Type of evaluation to run.
        system_prompt: System prompt for evaluation.
        user_prompt: User prompt for evaluation.
        n_evaluations: Number of evaluation rounds.
        eval_temperature: Temperature for evaluation rounds.

    Returns:
        GPTResult containing list of valid evaluations only.
    """
    eval_tasks = [
        client.run(
            eval_type,
            system_prompt,
            user_prompt,
            temperature=eval_temperature,
        )
        for _ in range(n_evaluations)
    ]

    all_evals = gpt_sequence(await asyncio.gather(*eval_tasks))
    return all_evals.map(
        lambda evals: [eval_ for eval_ in evals if eval_ and eval_.is_valid()]
    )


def _handle_no_valid_evaluations(
    paper: PaperWithRelatedSummary,
    cost: float,
) -> GPTResult[PaperResult]:
    """Handle the case where no valid evaluations were obtained.

    Args:
        paper: The paper being evaluated.
        cost: The cost from the failed evaluation attempts.

    Returns:
        GPTResult with error PaperResult.
    """
    logger.warning(f"Paper '{paper.title}': no valid evaluation results")
    return GPTResult(
        result=PaperResult.from_s2peer(
            paper=paper.paper.paper,
            y_pred=0,
            rationale_pred=RATIONALE_ERROR,
            structured_evaluation=None,
        ),
        cost=cost,
    )


def _handle_ensemble_evaluations(
    paper: PaperWithRelatedSummary,
    valid_evals: GPTResult[Sequence[GPTUncertain | GPTStructuredRaw | GPTFull]],
    target_mode: TargetMode,
) -> GPTResult[PaperResult]:
    """Handle evaluations with ensemble voting.

    Only values of types in `GPTPreConfidence` are used. The other types don't support
    a `confidence` field, so they are discarded. In practice, only GPTStructuredRaw and
    GPTFull will be passed to this.

    Args:
        paper: The paper being evaluated.
        valid_evals: Valid evaluation results.
        target_mode: Target mode for rating adjustment.

    Returns:
        GPTResult with ensemble PaperResult.
    """
    # Filter to only valid evaluations for ensemble
    structured_evaluations = valid_evals.map(
        lambda evals: [e for e in evals if isinstance(e, GPTPreConfidence)]
    )
    if not structured_evaluations.result:
        raise ValueError("No valid evaluations for ensemble aggregation.")

    # Aggregate evaluations using majority voting and TF-IDF
    aggregated_eval = structured_evaluations.map(aggregate_ensemble_evaluations)
    fixed_label = fix_evaluated_rating(aggregated_eval.result, target_mode).label

    return aggregated_eval.map(
        lambda s: PaperResult.from_s2peer(
            paper=paper.paper.paper,
            y_pred=fixed_label,
            rationale_pred=s.rationale,
            structured_evaluation=s if isinstance(s, GPTStructured) else None,
            confidence=s.confidence,
        )
    )


async def _extract_graph(
    client: LLMClient,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    paper: PaperWithRelatedSummary,
) -> GPTResult[Graph]:
    """Extract graph from paper if needed by the evaluation prompt.

    Args:
        client: LLM client for API calls.
        eval_prompt: Evaluation prompt template to check if graph is needed.
        graph_prompt: Graph prompt template for extraction.
        paper: Paper with related papers and summaries.

    Returns:
        GPTResult containing the extracted graph or empty graph.
    """
    if "graph" in eval_prompt.name:
        graph_prompt_text = format_graph_template(graph_prompt, paper.paper)
        graph_system_prompt = graph_prompt.system
        graph_result = await client.run(
            get_graph_type(graph_prompt.type_name),
            graph_system_prompt,
            graph_prompt_text,
        )
        graph = graph_result.map(
            lambda r: r.to_graph(title=paper.title, abstract=paper.abstract)
            if r
            else Graph.empty()
        )

        if graph.result.is_empty():
            logger.warning(f"Paper '{paper.title}': invalid Graph")
        return graph
    else:
        return gpt_unit(Graph.empty())


async def evaluate_paper(
    client: LLMClient,
    paper: PaperWithRelatedSummary,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    demonstrations: str,
    linearisation_method: LinearisationMethod,
    n_evaluations: int,
    eval_temperature: float,
    sources: set[PaperSource],
) -> GPTResult[PromptResult[GraphResult]]:
    """Evaluate a single paper's novelty using graph extraction and related papers.

    Args:
        client: LLM client for API calls.
        paper: Paper with related papers and summaries.
        eval_prompt: Prompt template for evaluation.
        graph_prompt: Prompt template for graph extraction.
        demonstrations: Text of demonstrations for few-shot prompting.
        linearisation_method: How to convert graph to text.
        n_evaluations: Number of evaluation rounds per paper for ensemble voting.
        eval_temperature: Temperature for evaluation rounds.
        sources: Which related paper sources to include.

    Returns:
        GraphResult with evaluation wrapped in PromptResult and GPTResult.
    """
    # Extract graph if needed
    graph = await _extract_graph(client, eval_prompt, graph_prompt, paper)

    # Prepare evaluation prompt
    eval_prompt_text = format_eval_template(
        eval_prompt, paper, graph.result, demonstrations, linearisation_method, sources
    )

    if eval_prompt.type_name == "GPTUncertain":
        eval_type, target_mode = GPTUncertain, TargetMode.UNCERTAIN
    elif eval_prompt.type_name == "GPTStructured":
        eval_type, target_mode = GPTStructuredRaw, TargetMode.BIN
    else:
        eval_type, target_mode = GPTFull, TargetMode.BIN

    if eval_type not in (GPTStructuredRaw, GPTFull):
        n_evaluations = 1

    # We want deterministic results if we're not doing multi-sampling
    if n_evaluations == 1:
        eval_temperature = 0

    valid_evals = await _run_evaluation_rounds(
        client,
        eval_type,
        eval_prompt.system,
        eval_prompt_text,
        n_evaluations,
        eval_temperature,
    )

    if not valid_evals.result:
        paper_result = _handle_no_valid_evaluations(paper, valid_evals.cost)
    elif eval_type in (GPTStructuredRaw, GPTFull):
        # Handle evaluations with ensemble
        paper_result = _handle_ensemble_evaluations(paper, valid_evals, target_mode)
    else:
        # Single evaluation (GPTUncertain)
        paper_result = valid_evals.map(lambda evals: evals[0]).map(
            lambda e: PaperResult.from_s2peer(
                paper=paper.paper.paper,
                y_pred=e.label,
                rationale_pred=e.rationale,
            )
        )

    return graph.lift(
        paper_result,
        lambda g, r: PromptResult(
            item=GraphResult.from_annotated(
                annotated=paper,
                result=r,
                graph=g,
            ),
            prompt=eval_prompt.with_user(eval_prompt_text),
        ),
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
    method: LinearisationMethod = LinearisationMethod.TOPO,
    sources: set[PaperSource] | None = None,
) -> str:
    """Format evaluation template using the paper graph and PETER-queried related papers."""
    if sources is None:
        sources = set(PaperSource)

    related = [p for p in paper.related if p.source in sources]
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        demonstrations=demonstrations,
        positive=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(method),
        approval=paper.paper.paper.approval,
    )


def format_related(related: Iterable[PaperRelatedSummarised]) -> str:
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
