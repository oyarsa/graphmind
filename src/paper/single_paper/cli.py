"""CLI interface and display functionality for single paper processing.

This module handles:
- Command-line interface for the single_paper command
- Interactive paper selection from search results
- Result display and formatting
- Progress tracking and user feedback
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import related_papers as rp
from paper.orc.arxiv_api import ArxivResult, arxiv_id_from_url
from paper.orc.download import parse_arxiv_latex
from paper.orc.latex_parser import SentenceSplitter
from paper.single_paper.paper_retrieval import fetch_s2_paper_info, search_arxiv_papers
from paper.single_paper.pipeline import QueryType, process_paper_from_query
from paper.util import arun_safe, atimer, seqcat, setup_logging
from paper.util.rate_limiter import get_limiter
from paper.util.serde import save_data

if TYPE_CHECKING:
    import arxiv  # type: ignore

    from paper.util.rate_limiter import Limiter

logger = logging.getLogger(__name__)


def filter_related(
    result: gpt.GraphResult, pol: pr.ContextPolarity, src: rp.PaperSource
) -> list[gpt.PaperRelatedSummarised]:
    """Filter related papers by polarity and source."""
    if not result.related:
        return []

    return [
        r
        for r in result.related
        if r.source.value == src.value and r.polarity.value == pol.value
    ]


def display_related_paper(related: gpt.PaperRelatedSummarised) -> str:
    """Display summary of related paper."""
    out = [
        f"    • {related.title}",
        f"      Score: {related.score:.3f}",
        f"      Summary: {related.summary[:100]}...",
    ]
    return "\n".join(out) + "\n"


def display_graph_results(result: gpt.GraphResult) -> None:
    """Display comprehensive results from processed paper with graph.

    Prints a formatted summary of the paper processing results including:
    - Main paper details (title, abstract, key terms, background, target)
    - Novelty evaluation results
    - Graph information
    - Citation-based related papers (positive and negative)
    - Semantic-based related papers (positive and negative)
    - Summaries and scores for each related paper

    Args:
        result: Complete processed paper with graph and evaluation.
    """
    print("\n" + "=" * 80)
    print("📊 COMPLETE PAPER PROCESSING RESULTS")
    print("=" * 80)

    # Print main paper information
    print(f"📑 Title: {result.paper.title}")
    print(f"📝 Abstract: {result.paper.abstract}")
    if result.terms:
        key_terms = seqcat(result.terms.methods, result.terms.tasks)
        print(f"🏷️  Key Terms: {', '.join(key_terms)}")
    if result.background:
        print(f"🎯 Background: {result.background}")
    if result.target:
        print(f"🚀 Target: {result.target}")
    print(f"📊 S2 References: {len(result.paper.references)}")
    print()

    # Print related papers summary
    if result.related:
        print(f"🔗 RELATED PAPERS ({len(result.related)} total):")
        print("-" * 50)
    else:
        print("🔗 RELATED PAPERS (0 total):")
        print("-" * 50)

    print("\n📖 CITATION-BASED PAPERS:")

    print("\n  ✅ Positive citations:")
    citations_positive = filter_related(
        result, pr.ContextPolarity.POSITIVE, rp.PaperSource.CITATIONS
    )
    print("\n".join(map(display_related_paper, citations_positive)))

    print("\n  ❌ Negative citations:")
    citations_negative = filter_related(
        result, pr.ContextPolarity.NEGATIVE, rp.PaperSource.CITATIONS
    )
    print("\n".join(map(display_related_paper, citations_negative)))

    print("\n🔍 SEMANTIC-BASED PAPERS:")

    print("\n  ✅ Positive semantic matches:")
    semantic_positive = filter_related(
        result, pr.ContextPolarity.POSITIVE, rp.PaperSource.SEMANTIC
    )
    print("\n".join(map(display_related_paper, semantic_positive)))

    print("\n  ❌ Negative semantic matches:")
    semantic_negative = filter_related(
        result, pr.ContextPolarity.NEGATIVE, rp.PaperSource.SEMANTIC
    )
    print("\n".join(map(display_related_paper, semantic_negative)))


async def process_paper(
    query: str,
    type_: QueryType,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
    detail: bool,
    eval_prompt: str,
    graph_prompt: str,
    demonstrations: str,
    demo_prompt: str,
    interactive: bool,
    output_file: Path | None,
) -> None:
    """Process a paper by title and display results.

    Convenience function that combines processing and display:
    1. Retrieves and processes paper through complete PETER pipeline.
    2. Displays comprehensive results to stdout.

    Args:
        query: Paper title or arXiv ID/URL to search for and process.
        type_: Whether to query arXiv by title or ID/URL.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        llm_model: GPT/Gemini model to use for all LLM API calls.
        encoder_model: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        detail: Show detailed paper information.
        eval_prompt: User prompt for paper evaluation.
        graph_prompt: User prompt for graph extraction.
        demonstrations: Demonstrations file for few-shot prompting.
        demo_prompt: Demonstration prompt key.
        interactive: If True, enables interactive paper selection for title searches.
        output_file: JSON file to store the full output.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv, or if no
            recommended papers or valid references are found during processing.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """

    limiter = get_limiter(1, 1)  # 1 request per second
    encoder = emb.Encoder(encoder_model)

    result = await atimer(
        process_paper_from_query(
            query,
            type_,
            top_k_refs,
            num_recommendations,
            num_related,
            llm_model,
            encoder,
            seed,
            limiter,
            eval_prompt,
            graph_prompt,
            demonstrations,
            demo_prompt,
            interactive,
        ),
        1,
    )

    graph_result = result.result
    print(f"✅ Found paper: {graph_result.paper.title} ({graph_result.paper.arxiv_id})")
    print(f"📄 Abstract: {graph_result.paper.abstract[:200]}...")
    print(f"📚 References: {len(graph_result.paper.references)}")
    print(f"📖 Sections: {len(graph_result.paper.sections)}")
    print()
    print("🚀 Processing through PETER pipeline and graph evaluation...")
    print(f"💰 Total cost: ${result.cost:.10f}")

    # Display novelty evaluation
    print(f"\n🎯 Novelty Evaluation: {graph_result.paper.label}")
    print(f"📝 Rationale: {graph_result.paper.rationale_pred[:1000]}")

    if graph_result.graph and not graph_result.graph.is_empty():
        print(f"\n📊 Graph extracted with {len(graph_result.graph.entities)} entities")

    if detail:
        display_graph_results(graph_result)

    if output_file:
        save_data(output_file, result)


def main(
    query: Annotated[
        str, typer.Argument(help="Title or arXiv ID/URL of the paper to process")
    ],
    type_: Annotated[
        QueryType, typer.Option("--type", help="Whether to query by title or arXiv.")
    ] = QueryType.TITLE,
    top_k_refs: Annotated[
        int, typer.Option(help="Number of top references to process by similarity")
    ] = 20,
    num_recommendations: Annotated[
        int, typer.Option(help="Number of recommended papers to fetch from S2 API")
    ] = 30,
    num_related: Annotated[
        int, typer.Option(help="Number of related papers per type (positive/negative)")
    ] = 2,
    llm_model: Annotated[
        str, typer.Option(help="GPT/Gemini model to use for API calls")
    ] = "gpt-4o-mini",
    encoder_model: Annotated[
        str, typer.Option(help="Embedding encoder model")
    ] = emb.DEFAULT_SENTENCE_MODEL,
    seed: Annotated[int, typer.Option(help="Random seed for GPT API calls")] = 0,
    detail: Annotated[
        bool, typer.Option(help="Show detailed paper information.")
    ] = False,
    eval_prompt: Annotated[
        str, typer.Option(help="User prompt for paper evaluation")
    ] = "full-graph-structured",
    graph_prompt: Annotated[
        str, typer.Option(help="User prompt for graph extraction")
    ] = "full",
    demonstrations: Annotated[
        str, typer.Option(help="Demonstrations file for few-shot prompting")
    ] = "orc_4",
    demo_prompt: Annotated[
        str, typer.Option(help="Demonstration prompt key")
    ] = "abstract",
    interactive: Annotated[
        bool,
        typer.Option(
            help="Interactive mode for paper selection (only for title search)"
        ),
    ] = False,
    output_file: Annotated[
        Path | None, typer.Option("--output", "-o", help="JSON file with full output.")
    ] = None,
) -> None:
    """Process a paper title through the complete PETER pipeline and print results."""
    setup_logging()
    dotenv.load_dotenv()
    arun_safe(
        process_paper,
        query,
        type_,
        top_k_refs,
        num_recommendations,
        num_related,
        llm_model,
        encoder_model,
        seed,
        detail,
        eval_prompt,
        graph_prompt,
        demonstrations,
        demo_prompt,
        interactive,
        output_file,
    )


async def get_paper_from_interactive_search(
    query: str, limiter: Limiter, api_key: str
) -> pr.Paper:
    """Search for papers interactively and process the selected one.

    Args:
        query: Search query string.
        limiter: Rate limiter for API requests.
        api_key: Semantic Scholar API key.

    Returns:
        Processed Paper object.

    Raises:
        ValueError: If no paper is selected or processing fails.
    """
    console = Console()

    # Search with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Searching arXiv for papers matching '{query}'...", total=None
        )
        results = await atimer(search_arxiv_papers(query), 3)
        progress.remove_task(task)

    if results is None:
        raise ValueError("Error searching arXiv")

    # Let user select
    selected = await atimer(
        asyncio.to_thread(select_paper_interactive, results, console), 3
    )
    if not selected:
        raise ValueError("No paper selected")

    console.print(f"[dim]📄 Paper: {selected.arxiv_title}[/dim]")
    console.print("\n[yellow]🚀 Processing through PETER pipeline...[/yellow]\n")

    # Continue with standard processing
    # Parse arXiv LaTeX
    sections, references = await atimer(
        asyncio.to_thread(parse_arxiv_latex, selected, SentenceSplitter()), 3
    )

    # Fetch S2 data
    s2_paper = await atimer(
        fetch_s2_paper_info(api_key, selected.arxiv_title, limiter), 3
    )

    if not s2_paper:
        raise ValueError(f"Paper not found on Semantic Scholar: {selected.arxiv_title}")

    return pr.Paper.from_s2(
        s2_paper, sections=sections, references=references, arxiv_id=selected.id
    )


def select_paper_interactive(
    results: list[arxiv.Result], console: Console
) -> ArxivResult | None:
    """Display search results and let user select a paper interactively.

    Args:
        results: List of arxiv.Result objects to display.
        console: Rich console for styled output.

    Returns:
        Selected ArxivResult or None if cancelled.
    """
    if not results:
        console.print(
            Panel(
                "[yellow]No papers found matching your search query.[/yellow]",
                title="No Results",
                border_style="yellow",
            )
        )
        return None

    table = Table(
        title=f"arXiv Search Results ({len(results)} papers found)",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )

    table.add_column("#", style="cyan", width=3)
    table.add_column("Title", style="bold", width=50)
    table.add_column("Authors", style="italic", width=30)
    table.add_column("Year", style="green", width=6)
    table.add_column("Abstract", style="dim", width=60)

    for i, result in enumerate(results, 1):
        year = result.published.year if result.published else "N/A"
        table.add_row(
            str(i),
            result.title,
            format_authors(result.authors, max_display=3),
            str(year),
            format_abstract(result.summary, max_length=200),
        )

    console.print(table)
    console.print()

    # Get user selection
    while True:
        selection = typer.prompt(
            f"Enter paper number (1-{len(results)}) or 'q' to quit",
            type=str,
        )

        if selection.lower() == "q":
            console.print("[blue]Selection cancelled.[/blue]")
            return None

        try:
            index = int(selection) - 1
            if 0 <= index < len(results):
                selected = results[index]
                console.print(
                    f"\n[green]✅ Selected paper:[/green] [bold]{selected.title}[/bold]"
                )
                return ArxivResult(
                    id=arxiv_id_from_url(selected.entry_id),
                    openreview_title=selected.title,
                    arxiv_title=selected.title,
                )
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(results)}.[/red]"
                )
        except ValueError:
            console.print("[red]Invalid input. Please enter a number or 'q'.[/red]")


def format_authors(authors: Sequence[arxiv.Result.Author], *, max_display: int) -> str:
    """Format author list for display.

    Args:
        authors: List of arxiv Author objects.
        max_display: Maximum number of authors to display before using "et al."

    Returns:
        Formatted author string.
    """
    if not authors:
        return "Unknown"

    author_names = [author.name for author in authors]
    if len(author_names) <= max_display:
        return ", ".join(author_names)
    else:
        return f"{', '.join(author_names[:max_display])} et al."


def format_abstract(abstract: str, *, max_length: int) -> str:
    """Format abstract for display by truncating if necessary.

    Args:
        abstract: Full abstract text.
        max_length: Maximum length before truncation.

    Returns:
        Formatted abstract string.
    """
    if not abstract:
        return "No abstract available"

    # Clean up whitespace
    abstract = " ".join(abstract.split())

    if len(abstract) <= max_length:
        return abstract

    return abstract[: max_length - 3] + "..."
