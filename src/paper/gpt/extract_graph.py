"""Generate graphs from papers from the PeerRead-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from paper import hierarchical_graph
from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.model import (
    Graph,
    PaperRelatedSummarised,
    PaperTerms,
    PaperWithRelatedSummary,
    Prompt,
    PromptResult,
)
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
