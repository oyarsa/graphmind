"""Generate graphs from papers from the PeerRead-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, override

from pydantic import BaseModel, ConfigDict, Field

from paper import hierarchical_graph
from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.model import (
    Entity,
    EntityType,
    Graph,
    Prompt,
    PromptResult,
    Relationship,
)
from paper.util.serde import Record, save_data

logger = logging.getLogger(__name__)


def _at[T](seq: Sequence[T], idx: int, desc: str) -> T | None:
    """Get `seq[idx]` if possible, otherwise return None and log warning with `desc`."""
    try:
        return seq[idx]
    except IndexError:
        logger.warning("Invalid index at '%s': %d out of %d", desc, idx, len(seq))
        return None


class GPTGraph(BaseModel):
    """Graph representing the paper."""

    title: Annotated[str, Field(description="Title of the paper.")]
    primary_area: Annotated[
        str,
        Field(
            description="The primary subject area of the paper picked from the ICLR list of"
            " topics."
        ),
    ]
    keywords: Annotated[
        Sequence[str],
        Field(description="Keywords that summarise the key aspects of the paper."),
    ]
    tldr: Annotated[str, Field(description="Sentence that summarises the paper.")]
    claims: Annotated[
        Sequence[ClaimEntity],
        Field(
            description="Main contributions the paper claims to make, with connections to"
            " target `methods`."
        ),
    ]
    methods: Annotated[
        Sequence[MethodEntity],
        Field(
            description="Methods used to verify the claims, with connections to target"
            " `experiments`"
        ),
    ]
    experiments: Annotated[
        Sequence[ExperimentEntity],
        Field(description="Experiments designed to put methods in practice."),
    ]

    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""

        # Track seen names to detect duplicates
        names_map: dict[tuple[str, EntityType], str] = {}
        names_seen: set[str] = set()

        def entity(name: str, type: EntityType) -> Entity:
            if name in names_seen:
                unique_name = f"{name} ({type.value})"
            else:
                names_seen.add(name)
                unique_name = name
            names_map[name, type] = unique_name
            return Entity(name=unique_name, type=type)

        entities = [
            entity(self.title, EntityType.TITLE),
            entity(self.primary_area, EntityType.PRIMARY_AREA),
            *(entity(kw, EntityType.KEYWORD) for kw in self.keywords),
            entity(self.tldr, EntityType.TLDR),
            *(entity(c.text, EntityType.CLAIM) for c in self.claims),
            *(entity(m.text, EntityType.METHOD) for m in self.methods),
            *(entity(x.text, EntityType.EXPERIMENT) for x in self.experiments),
        ]

        relationships = [
            Relationship(
                source=names_map[self.title, EntityType.TITLE],
                target=names_map[self.primary_area, EntityType.PRIMARY_AREA],
            ),
            *(
                Relationship(
                    source=names_map[self.title, EntityType.TITLE],
                    target=names_map[kw, EntityType.KEYWORD],
                )
                for kw in self.keywords
            ),
            Relationship(
                source=names_map[self.title, EntityType.TITLE],
                target=names_map[self.tldr, EntityType.TLDR],
            ),
            *(
                Relationship(
                    source=names_map[self.tldr, EntityType.TLDR],
                    target=names_map[c.text, EntityType.CLAIM],
                )
                for c in self.claims
            ),
            *(
                Relationship(
                    source=names_map[c.text, EntityType.CLAIM],
                    target=names_map[target.text, EntityType.METHOD],
                )
                for c in self.claims
                for midx in c.method_indices
                if (target := _at(self.methods, midx, "claim->method"))
            ),
            *(
                Relationship(
                    source=names_map[m.text, EntityType.METHOD],
                    target=names_map[target.text, EntityType.EXPERIMENT],
                )
                for m in self.methods
                for eidx in m.experiment_indices
                if (target := _at(self.experiments, eidx, "method->exp"))
            ),
        ]

        return Graph(
            title=title,
            abstract=abstract,
            entities=entities,
            relationships=relationships,
        )


class ClaimEntity(BaseModel):
    """Entity representing a claim made in the paper."""

    text: Annotated[str, Field(description="Description of a claim made by the paper")]
    method_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices for the `methods` connected to this claim in the `methods`"
            " list. There must be at least one connected `method`."
        ),
    ]


class MethodEntity(BaseModel):
    """Entity representing a method described in the paper to support the claims."""

    text: Annotated[
        str,
        Field(
            description="Description of a method used to validate claims from the paper."
        ),
    ]
    index: Annotated[
        int, Field(description="Index for this method in the `methods` list")
    ]
    experiment_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices for the `experiments` connected to this method in the "
            " `experiments` list. There must be at least one connected `experiment`."
        ),
    ]


class ExperimentEntity(BaseModel):
    """Entity representing an experiment used to validate a method from the paper."""

    text: Annotated[
        str,
        Field(
            description="Description of an experiment used to validate the methods from"
            " the paper."
        ),
    ]
    index: Annotated[
        int, Field(description="Index for this method in the `experiments` list")
    ]


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
        output_dir: Where the graph and image wll be persisted. The graph is saved as
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
        output_dir: Where the graph and image wll be persisted. The graph is saved as
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
