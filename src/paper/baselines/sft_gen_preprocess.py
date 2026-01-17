"""Preprocess papers by extracting graphs and formatting related papers for SFT-Gen.

This script enriches PaperWithRelatedSummary data with:
- Linearised knowledge graphs (extracted via GPT or loaded from cache)
- Formatted positive and negative related paper text

The output is PaperWithGraphContext, ready for use with sft_gen_graph.py.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from paper.gpt.prompts import PromptTemplate

from paper import peerread as pr
from paper.gpt.evaluate_paper_graph import format_related
from paper.gpt.graph_cache import (
    compute_cache_key,
    load_cached_graphs,
    save_graphs_to_cache,
)
from paper.gpt.model import (
    Graph,
    LinearisationMethod,
    PaperRelatedSummarised,
    PaperWithGraphContext,
    PaperWithRelatedSummary,
    PromptResult,
)
from paper.gpt.prompts.evaluate_graph import GRAPH_EVAL_USER_PROMPTS
from paper.gpt.prompts.extract_graph import GRAPH_EXTRACT_USER_PROMPTS
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_unit
from paper.types import PaperSource
from paper.util import (
    Timer,
    batch_map_with_progress,
    cli,
    dotenv,
    sample,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Preprocess papers with graph extraction for SFT-Gen training.",
)


def format_related_by_polarity(
    related: Iterable[PaperRelatedSummarised],
    polarity: pr.ContextPolarity,
) -> str:
    """Build prompt from related papers filtered by polarity."""
    return format_related(p for p in related if p.polarity is polarity)


async def extract_graph_for_paper(
    client: LLMClient,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    paper: PaperWithRelatedSummary,
    cached_graphs: dict[str, Graph],
) -> GPTResult[Graph]:
    """Extract graph for a single paper, using cache if available.

    This is a wrapper around evaluate_paper_graph.extract_graph that works
    without needing the eval_prompt to contain 'graph' in its name.
    """
    from paper.gpt.evaluate_paper_graph import format_graph_template
    from paper.gpt.graph_types import get_graph_type
    from paper.util import hashstr

    # Check cache first using paper's title+abstract hash
    paper_id = hashstr(paper.title + paper.abstract)
    if paper_id in cached_graphs:
        logger.debug("Using cached graph for '%s'", paper.title[:50])
        return gpt_unit(cached_graphs[paper_id])

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
        logger.warning("Paper '%s': invalid Graph", paper.title[:50])
    return graph


async def process_paper(
    client: LLMClient,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    paper: PaperWithRelatedSummary,
    cached_graphs: dict[str, Graph],
    linearisation_method: LinearisationMethod,
    sources: set[PaperSource],
) -> tuple[PaperWithGraphContext, Graph, float]:
    """Process a single paper to create enriched training data.

    Returns:
        Tuple of (enriched paper data, extracted graph, cost).
    """
    # Extract graph
    graph_result = await extract_graph_for_paper(
        client, eval_prompt, graph_prompt, paper, cached_graphs
    )
    graph = graph_result.result

    # Linearise graph
    graph_text = graph.to_text(linearisation_method) if not graph.is_empty() else ""

    # Filter related papers by source
    related = [p for p in paper.related if p.source in sources]

    # Format related papers by polarity
    positive_related = format_related_by_polarity(related, pr.ContextPolarity.POSITIVE)
    negative_related = format_related_by_polarity(related, pr.ContextPolarity.NEGATIVE)

    enriched = PaperWithGraphContext(
        paper=paper.paper,
        related=paper.related,
        graph_text=graph_text,
        positive_related=positive_related,
        negative_related=negative_related,
        linearisation_method=linearisation_method.value,
        sources=[s.value for s in sources],
    )

    return enriched, graph, graph_result.cost


async def preprocess_papers(
    papers: list[PaperWithRelatedSummary],
    client: LLMClient,
    eval_prompt: PromptTemplate,
    graph_prompt: PromptTemplate,
    cached_graphs: dict[str, Graph],
    linearisation_method: LinearisationMethod,
    sources: set[PaperSource],
    batch_size: int,
) -> tuple[list[PaperWithGraphContext], dict[str, Graph], float]:
    """Process all papers and return enriched data with new graphs.

    Returns:
        Tuple of (enriched papers, newly extracted graphs, total cost).
    """
    new_graphs: dict[str, Graph] = {}
    total_cost = 0.0

    async def process(
        paper: PaperWithRelatedSummary,
    ) -> tuple[PaperWithGraphContext, Graph, float]:
        return await process_paper(
            client,
            eval_prompt,
            graph_prompt,
            paper,
            cached_graphs,
            linearisation_method,
            sources,
        )

    results = await batch_map_with_progress(process, papers, batch_size, name="papers")

    enriched_papers: list[PaperWithGraphContext] = []
    for enriched, graph, cost in results:
        enriched_papers.append(enriched)
        total_cost += cost
        # Track newly extracted graphs (not from cache)
        if graph and not graph.is_empty() and graph.id not in cached_graphs:
            new_graphs[graph.id] = graph

    return enriched_papers, new_graphs, total_cost


@app.command(no_args_is_help=True)
def run(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input JSON file containing PaperWithRelatedSummary data.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for enriched data.",
        ),
    ],
    cache_dir: Annotated[
        Path,
        typer.Option(
            help="Directory for graph extraction cache.",
        ),
    ] = Path("output/.cache/graphs"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="GPT model for graph extraction.",
        ),
    ] = "gpt-4o-mini",
    graph_prompt: Annotated[
        str,
        typer.Option(
            help="Prompt key for graph extraction.",
            click_type=cli.Choice(GRAPH_EXTRACT_USER_PROMPTS),
        ),
    ] = "excerpts",
    linearisation: Annotated[
        LinearisationMethod,
        typer.Option(
            help="How to convert the extracted graph into text.",
        ),
    ] = LinearisationMethod.TOPO,
    sources: Annotated[
        list[PaperSource],
        typer.Option(
            help="Which related paper sources to include.",
        ),
    ] = [PaperSource.CITATIONS, PaperSource.SEMANTIC],  # noqa: B006
    seed: Annotated[
        int,
        typer.Option(
            help="Random seed for data shuffling and GPT calls.",
        ),
    ] = 0,
    temperature: Annotated[
        float,
        typer.Option(
            help="Temperature for graph extraction GPT calls.",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Number of papers to process per batch.",
        ),
    ] = 100,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Limit number of papers to process.",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Force graph regeneration, ignoring cache.",
        ),
    ] = False,
) -> None:
    """Preprocess papers by extracting graphs and formatting related papers.

    Reads PaperWithRelatedSummary data, extracts knowledge graphs using GPT,
    and outputs PaperWithGraphContext ready for SFT-Gen training.
    """
    asyncio.run(
        _preprocess(
            input_file=input_file,
            output_file=output_file,
            cache_dir=cache_dir,
            model=model,
            graph_prompt_key=graph_prompt,
            linearisation_method=linearisation,
            sources=set(sources),
            seed=seed,
            temperature=temperature,
            batch_size=batch_size,
            limit=limit,
            no_cache=no_cache,
        )
    )


async def _preprocess(
    input_file: Path,
    output_file: Path,
    cache_dir: Path,
    model: str,
    graph_prompt_key: str,
    linearisation_method: LinearisationMethod,
    sources: set[PaperSource],
    seed: int,
    temperature: float,
    batch_size: int,
    limit: int | None,
    no_cache: bool,
) -> None:
    """Async implementation of preprocessing."""
    dotenv.load_dotenv()

    rng = random.Random(seed)

    # Load input data
    logger.info("Loading input data from %s", input_file)
    papers = sample(
        PromptResult.unwrap(
            load_data(input_file, PromptResult[PaperWithRelatedSummary])
        ),
        limit,
        rng,
    )
    logger.info("Loaded %d papers", len(papers))

    # Set up graph prompt
    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    if not graph_prompt.system:
        raise ValueError(
            f"Graph prompt {graph_prompt.name!r} does not have a system prompt."
        )

    # Use a dummy eval prompt - we only need graph extraction
    eval_prompt = GRAPH_EVAL_USER_PROMPTS["full-graph-structured"]

    # Load graph cache
    cached_graphs: dict[str, Graph] = {}
    cache_key = compute_cache_key(
        model=model,
        temperature=temperature,
        graph_prompt_key=graph_prompt_key,
        seed=seed,
        input_file=input_file,
    )
    cache_path = cache_dir / cache_key

    if not no_cache and (loaded_graphs := load_cached_graphs(cache_path)):
        cached_graphs = dict(loaded_graphs)

    # Set up LLM client
    client = LLMClient.new_env(model=model, seed=seed, temperature=temperature)

    # Adjust batch size for Gemini
    if "gemini" in model:
        batch_size = min(batch_size, 25)

    # Process papers
    with Timer() as timer:
        enriched_papers, new_graphs, total_cost = await preprocess_papers(
            papers=papers,
            client=client,
            eval_prompt=eval_prompt,
            graph_prompt=graph_prompt,
            cached_graphs=cached_graphs,
            linearisation_method=linearisation_method,
            sources=sources,
            batch_size=batch_size,
        )

    logger.info("Time elapsed: %s", timer.human)
    logger.info("Total cost: $%.10f", total_cost)
    logger.info("Calls made: %d", client.calls_made)
    logger.info("Tokens used: %d", client.tokens_used)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_data(output_file, enriched_papers)
    logger.info("Saved %d enriched papers to %s", len(enriched_papers), output_file)

    # Update graph cache
    if new_graphs:
        all_graphs = cached_graphs | new_graphs
        cache_params = {
            "model": model,
            "temperature": str(temperature),
            "graph_prompt_key": graph_prompt_key,
            "seed": str(seed),
            "input_file": str(input_file),
        }
        save_graphs_to_cache(cache_path, all_graphs, cache_params)
        logger.info("Updated cache with %d new graphs", len(new_graphs))


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


if __name__ == "__main__":
    app()
