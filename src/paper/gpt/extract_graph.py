"""Generate graphs from papers from the PeerRead-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override

from pydantic import BaseModel, ConfigDict

from paper import hierarchical_graph
from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.model import Graph, Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate
from paper.util import read_resource
from paper.util.serde import Record, save_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class GraphPrompt(PromptTemplate):
    """Graph prompt loaded from a file. Includes the output type."""

    type_name: str


def load_graph_prompts(name: str) -> Mapping[str, GraphPrompt]:
    """Load graph prompts from a TOML file in the prompts package.

    Args:
        name: Name of the TOML file in `paper.gpt.prompts`, without extension.

    Returns:
        Dictionary mapping prompt names to their text content.
    """
    text = read_resource("gpt.prompts", f"{name}.toml")
    return {
        p["name"]: GraphPrompt(
            name=p["name"],
            system=p.get("system", ""),
            template=p["prompt"],
            type_name=p["type"],
        )
        for p in tomllib.loads(text)["prompts"]
    }


class GraphResult(Record):
    """Extracted graph and paper evaluation results."""

    graph: Graph
    paper: PaperResult

    @property
    @override
    def id(self) -> str:
        return self.paper.id


def save_graphs(
    graph_results: Iterable[PromptResult[GraphResult]], output_dir: Path
) -> None:
    """Save results as a JSON file with the prompts and graphs in GraphML format.

    Args:
        graph_results: Result of paper evaluation with graphs, wrapped with prompt.
        output_dir: Where the graph and image will be persisted. The graph is saved as
            GraphML and the image as PNG.
    """

    class Output(BaseModel):
        model_config = ConfigDict(frozen=True)

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
