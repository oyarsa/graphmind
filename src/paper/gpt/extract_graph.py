"""Generate graphs from papers from the PeerRead-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import asyncio
import logging
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rich.console import Console
from rich.table import Table

from paper import hierarchical_graph
from paper.gpt.evaluate_paper_graph import CLASSIFY_USER_PROMPTS, evaluate_graphs
from paper.gpt.model import (
    Entity,
    EntityType,
    Graph,
    Paper,
    PaperGraph,
    Prompt,
    PromptResult,
    Relationship,
    graph_to_digraph,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
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
    read_resource,
    setup_logging,
)
from paper.util.serde import load_data

logger = logging.getLogger(__name__)


class IndexedEntity(BaseModel):
    """Entity from the paper. It belongs to a list and carries its index in that list."""

    model_config = ConfigDict(frozen=True)

    index: Annotated[
        int, Field(description="Index of this entity in its original list.")
    ]
    text: Annotated[
        str, Field(description="Sentence from the paper describing this entity.")
    ]


class ConnectedEntity(IndexedEntity):
    """Entity from a paper that has an index and is connected to other entities by index.

    The source indices from other entities are from the original list containing the
    connected entities.
    """

    source_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices of the entities connected to this one in their original"
            " list."
        ),
    ]


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


_GRAPH_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)
_PRIMARY_AREAS: list[str] = tomllib.loads(
    read_resource("gpt.prompts", "primary_areas.toml")
)["primary_areas"]
_GRAPH_USER_PROMPTS = load_prompts("extract_graph")


async def _generate_graph(
    client: AsyncOpenAI,
    example: Paper,
    model: str,
    user_prompt: PromptTemplate,
    *,
    seed: int,
) -> GPTResult[PromptResult[Graph]]:
    user_prompt_text = user_prompt.template.format(
        title=example.title,
        abstract=example.abstract,
        main_text=example.main_text(),
        primary_areas=", ".join(_PRIMARY_AREAS),
    )
    result = await run_gpt(
        GPTGraph,
        client,
        _GRAPH_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )
    graph = (
        result.result.to_graph(title=example.title, abstract=example.abstract)
        if result.result
        else Graph(
            title=example.title,
            abstract=example.abstract,
            entities=[],
            relationships=[],
        )
    )
    logger.debug(graph)
    return GPTResult(
        result=PromptResult(
            item=graph,
            prompt=Prompt(user=user_prompt_text, system=_GRAPH_SYSTEM_PROMPT),
        ),
        cost=result.cost,
    )


async def _generate_graphs(
    client: AsyncOpenAI,
    data: list[Paper],
    model: str,
    user_prompt: PromptTemplate,
    output_intermediate_path: Path,
    *,
    seed: int,
) -> GPTResult[list[PromptResult[Graph]]]:
    total_cost = 0
    graph_results: list[PromptResult[Graph]] = []

    tasks = [
        _generate_graph(client, example, model, user_prompt, seed=seed)
        for example in data
    ]

    for task in progress.as_completed(tasks, desc="Extracting graphs"):
        result = await task
        total_cost += result.cost

        graph_results.append(result.result)
        append_intermediate_result(Graph, output_intermediate_path, result.result)

    return GPTResult(graph_results, total_cost)


def _save_graphs(
    paper_graphs: Iterable[PaperGraph],
    output_dir: Path,
) -> None:
    """Save results as a JSON file with the prompts and graphs in GraphML format.

    Args:
        paper_graphs: Papers and the graphs generated from them.
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
            paper=pg.paper.title,
            graphml=graph_to_digraph(pg.graph.item).graphml(),
            graph=pg.graph.item,
            prompt=pg.graph.prompt,
        )
        for pg in paper_graphs
    )
    (output_dir / "result_graphs.json").write_bytes(
        TypeAdapter(list[Output]).dump_json(output, indent=2)
    )


def _display_graphs(
    model: str,
    paper_graphs: Iterable[PaperGraph],
    graph_user_prompt_key: str,
    output_dir: Path,
    display_gui: bool,
) -> None:
    """Plot graphs to PNG files and (optionally) the screen.

    Args:
        model: GPT model used to generate the Graph
        graph_user_prompt_key: Key to the prompt used to generate the Graph
        paper_graphs: Papers and the graphs generated from them
        output_dir: Where the graph and image wll be persisted. The graph is saved as
            GraphML and the image as PNG.
        display_gui: If True, show the graph on screen. This suspends the process until
            the plot is closed.
    """
    for pg in paper_graphs:
        dag = graph_to_digraph(pg.graph.item)

        try:
            dag.visualise_hierarchy(
                img_path=output_dir / f"{pg.paper.title}.png",
                display_gui=display_gui,
                description=f"index - model: {model} - prompt: {graph_user_prompt_key}\n"
                f"status: {pg.graph.item.valid_status}\n",
            )
        except hierarchical_graph.GraphError:
            logger.exception("Error visualising graph")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    data_path: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containig the papers data."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The path to the output directory where files will be saved."
        ),
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="The model to use for the extraction.")
    ] = "gpt-4o-mini",
    limit: Annotated[
        int | None, typer.Option(help="Limit to the number of papers to process.")
    ] = None,
    graph_user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for the graph extraction.",
            click_type=cli.choice(_GRAPH_USER_PROMPTS),
        ),
    ] = "simple",
    classify_user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for paper classification.",
            click_type=cli.choice(CLASSIFY_USER_PROMPTS),
        ),
    ] = "simple",
    display: Annotated[bool, typer.Option(help="Display the extracted graph")] = False,
    classify: Annotated[
        bool, typer.Option(help="Classify the papers based on the extracted entities.")
    ] = False,
    continue_papers: Annotated[
        Path | None,
        typer.Option(help="Path to file with data from a previous run."),
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results."),
    ] = False,
    seed: Annotated[int, typer.Option(help="Seed to set in the OpenAI call.")] = 0,
) -> None:
    """Extract graphs from the papers in the dataset and (optionally) classify them."""
    asyncio.run(
        extract_graph(
            model,
            data_path,
            limit,
            graph_user_prompt,
            classify_user_prompt,
            display,
            output_dir,
            classify,
            continue_papers,
            continue_,
            seed,
        )
    )


async def extract_graph(
    model: str,
    data_path: Path,
    limit: int | None,
    graph_user_prompt_key: str,
    classify_user_prompt_key: str,
    display: bool,
    output_dir: Path,
    classify: bool,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
) -> None:
    """Extract graphs from the papers in the dataset and (maybe) classify them.

    See `gpt.models.Graph` for the graph structure and rules.

    The papers should come from the PeerRead-Review dataset as processed by the
    paper.peerread module.

    The classification part is optional. It uses the generated graphs as input as saves
    the results (metrics and predictions) to {output_dir}/classification.

    Args:
        model: GPT model code. Must support Structured Outputs.
        data_path: Path to the JSON file containing the input papers data.
        limit: Number of papers to process. Defaults to 1 example. If None, process all.
        graph_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_GRAPH_USER_PROMPTS` for available options or `_display_prompts` for more.
        classify_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_CLASSIFY_USER_PROMPTS` for available options or `_display_prompts` for more.
        display: If True, show each graph on screen. This suspends the process until
            the plot is closed.
        output_dir: Directory to save the output files: serialised graphs (GraphML),
            plot images (PNG) and classification results (JSON), if classification is
            enabled.
        classify: If True, classify the papers based on the generated graph.
        continue_papers_file: If provided, check for entries in the input data. If they
            are there, we use those results and skip processing them.
        continue_: Use data fromn `continue_papers`.
        seed: Seed for the OpenAI API call

    Returns:
        None. The output is saved to disk.
    """
    logger.info(display_params())

    dotenv.load_dotenv()

    if model not in MODELS_ALLOWED:
        raise ValueError(f"Model {model} not in allowed models: {MODELS_ALLOWED}")
    model = MODEL_SYNONYMS.get(model, model)

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    data = load_data(data_path, Paper)

    papers = data[:limit]
    graph_user_prompt = _GRAPH_USER_PROMPTS[graph_user_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        Graph, output_intermediate_file, continue_papers_file, papers, continue_
    )
    if not papers_remaining.remaining:
        logger.info("No items left to process. They're all on the `continues` file.")
    elif continue_:
        logger.info(
            "Skipping %d items from the `continue` file.", len(papers_remaining.done)
        )

    with Timer() as timer_gen:
        graph_results = await _generate_graphs(
            client,
            papers_remaining.remaining,
            model,
            graph_user_prompt,
            output_intermediate_file,
            seed=seed,
        )

    logger.info(f"Graph generation time elapsed: {timer_gen.human}")
    logger.info(f"Total graph generation cost: ${graph_results.cost:.10f}")

    graph_results_all = graph_results.result + papers_remaining.done
    paper_graphs = [
        PaperGraph(
            paper=p, graph=next(g for g in graph_results_all if g.item.id == p.id)
        )
        for p in papers
    ]

    _save_graphs(paper_graphs, output_dir)
    _display_graphs(model, paper_graphs, graph_user_prompt_key, output_dir, display)
    _display_validation(graph_results_all)

    if classify:
        await evaluate_graphs(
            client,
            model,
            paper_graphs,
            classify_user_prompt_key,
            output_dir,
            # We always want new paper classifications after processing the graphs
            continue_papers_file=None,
            continue_=True,
            seed=seed,
        )


def _display_validation(results: Iterable[PromptResult[Graph]]) -> None:
    valids: defaultdict[str, list[Graph]] = defaultdict(list)
    for x in results:
        valids[x.item.valid_status].append(x.item)
    valid_items = sorted(valids.items(), key=lambda x: len(x[1]))

    valid_table = Table("Validation message", "Count", "Example (title)")
    for msg, graphs in valid_items:
        valid_table.add_row(f"«{msg}»", str(len(graphs)), graphs[0].title)

    Console().print(valid_table)


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    items = [
        ("GRAPH EXTRACTION PROMPTS", _GRAPH_USER_PROMPTS),
        ("CLASSIFICATION PROMPTS", CLASSIFY_USER_PROMPTS),
    ]
    for title, prompts in items:
        print_prompts(title, prompts, detail=detail)
        print()
