"""Generate graphs from papers from the PeerRead-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field

from paper import hierarchical_graph
from paper.gpt.evaluate_paper import GPTStructured, PaperResult, fix_evaluated_rating
from paper.gpt.graph_types.base import GPTGraphBase
from paper.gpt.model import (
    Graph,
    PaperRelatedSummarised,
    PaperTerms,
    PaperWithRelatedSummary,
    Prompt,
    PromptResult,
)
from paper.gpt.run_gpt import GPTResult, LLMClient
from paper.types import Immutable, PaperProxy
from paper.util.serde import save_data

logger = logging.getLogger(__name__)


class GraphResult(Immutable, PaperProxy[PaperResult]):
    """Extracted graph and paper evaluation results."""

    graph: Graph
    paper: PaperResult
    related: Sequence[PaperRelatedSummarised] | None = None

    # Optional annotation data from the original PeerReadAnnotated paper
    # These will be None for papers processed with the old pipeline
    terms: PaperTerms | None = None
    background: str | None = None
    target: str | None = None

    @property
    def rationale_pred(self) -> str:
        """Predicted rationale from the underlying paper result."""
        return self.paper.rationale_pred

    @classmethod
    def from_annotated(
        cls,
        annotated: PaperWithRelatedSummary,
        result: PaperResult,
        graph: Graph,
    ) -> Self:
        """Create GraphResult with annotation data."""
        return cls(
            graph=graph,
            paper=result,
            related=annotated.related,
            terms=annotated.terms,
            background=annotated.background,
            target=annotated.target,
        )

    @classmethod
    def from_paper(
        cls,
        paper: PaperResult,
        graph: Graph,
        related: Sequence[PaperRelatedSummarised] | None = None,
    ) -> Self:
        """Create GraphResult without annotation data (for backward compatibility)."""
        return cls(graph=graph, paper=paper, related=related)


class ExtractedGraph(Immutable, PaperProxy[PaperWithRelatedSummary]):
    """Extracted graph with the original paper."""

    graph: Graph
    paper: PaperWithRelatedSummary


def construct_graph_result(
    paper: PaperWithRelatedSummary, graph: Graph, evaluation: GPTStructured
) -> GraphResult:
    """Construct the final graph result from components.

    Args:
        paper: Paper with related papers and summaries.
        graph: Extracted graph representation.
        evaluation: Novelty evaluation result.

    Returns:
        Complete GraphResult.
    """
    result = PaperResult.from_s2peer(
        paper=paper.paper.paper,
        y_pred=fix_evaluated_rating(evaluation).label,
        rationale_pred=evaluation.rationale,
        structured_evaluation=evaluation,
    )
    return GraphResult.from_annotated(annotated=paper, graph=graph, result=result)


class EvaluationResult(Immutable):
    """Evaluation result with cost."""

    result: Annotated[GraphResult, Field(description="Evaluated graph result.")]
    cost: Annotated[float, Field(description="Total cost of using the LLM API.")]

    @classmethod
    def from_(cls, result: GPTResult[GraphResult]) -> Self:
        """Create EvaluationResult from GPTResult+GraphResult."""
        return cls(result=result.result, cost=result.cost)


async def extract_graph_core(
    client: LLMClient,
    graph_type: type[GPTGraphBase],
    system_prompt: str,
    user_prompt: str,
    title: str,
    abstract: str,
) -> GPTResult[Graph]:
    """Run LLM graph extraction, map to Graph, and warn if empty.

    This is the shared core used by both the single-paper live path and the
    batch experiment path.

    Args:
        client: LLM client for API calls.
        graph_type: Pydantic graph output type (e.g. GPTExcerpt).
        system_prompt: System prompt for graph extraction.
        user_prompt: Formatted user prompt for graph extraction.
        title: Paper title (used for the Graph metadata).
        abstract: Paper abstract (used for the Graph metadata).

    Returns:
        GPTResult containing the extracted Graph.
    """
    result = await client.run(graph_type, system_prompt, user_prompt)
    graph = result.map(
        lambda r: r.to_graph(title=title, abstract=abstract) if r else Graph.empty()
    )
    if graph.result.is_empty():
        logger.warning(f"Paper '{title}': invalid Graph")
    return graph


def save_graphs(
    graph_results: Iterable[PromptResult[GraphResult]], output_dir: Path
) -> None:
    """Save results as a JSON file with the prompts and graphs in GraphML format.

    Args:
        graph_results: Result of paper evaluation with graphs, wrapped with prompt.
        output_dir: Where the graph and image will be persisted. The graph is saved as
            GraphML and the image as PNG.
    """

    class Output(Immutable):
        paper: str
        graphml: str
        graph: Graph
        prompt: Prompt

    output: list[Output] = []

    output.extend(
        Output(
            paper=gr.item.paper.title,
            graphml=gr.item.graph.to_digraph().graphml(),
            graph=gr.item.graph,
            prompt=gr.prompt,
        )
        for gr in graph_results
    )
    save_data(output_dir / "result_graphs.json", output)


def display_graphs(
    model: str,
    graph_results: Iterable[GraphResult],
    graph_user_prompt_key: str,
    output_dir: Path,
    display_gui: bool,
) -> None:
    """Plot graphs to PNG files and (optionally) the screen.

    Args:
        model: GPT model used to generate the Graph
        graph_user_prompt_key: Key to the prompt used to generate the Graph
        graph_results: Result of paper evaluation with graphs.
        output_dir: Where the graph and image will be persisted. The graph is saved as
            GraphML and the image as PNG.
        display_gui: If True, show the graph on screen. This suspends the process until
            the plot is closed.
    """
    for pg in graph_results:
        try:
            pg.graph.to_digraph().visualise_hierarchy(
                img_path=output_dir / f"{pg.paper.title}.png",
                display_gui=display_gui,
                description=f"index - model: {model} - prompt: {graph_user_prompt_key}\n"
                f"status: {pg.graph.valid_status}\n",
            )
        except hierarchical_graph.GraphError:
            logger.exception("Error visualising graph")
