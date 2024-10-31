"""Generate graphs from papers from the ASAP-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import tomllib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import override

import dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rich.console import Console
from rich.table import Table

from paper import hierarchical_graph
from paper.gpt.evaluate_graph import (
    CLASSIFY_USER_PROMPTS,
    evaluate_graphs,
)
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
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    append_intermediate_result,
    get_id,
    get_remaining_items,
    run_gpt,
)
from paper.progress import as_completed
from paper.util import Timer, read_resource, setup_logging

logger = logging.getLogger("paper.gpt.extract_graph")


class GPTGraphBase(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def to_graph(self, title: str, abstract: str) -> Graph: ...


class GPTGraphStrict(GPTGraphBase):
    """Graph representing the paper."""

    title: str = Field(description="Title of the paper.")
    primary_area: str = Field(
        description="The primary subject area of the paper picked from the ICLR list of"
        " topics."
    )
    keywords: Sequence[str] = Field(
        description="Keywords that summarise the key aspects of the paper."
    )
    tldr: str = Field(description="Sentence that summarises the paper.")
    claims: Sequence[IndexedEntity] = Field(
        description="Main contributions the paper claims to make."
    )
    methods: Sequence[ConnectedEntity] = Field(
        description="Methods used to verify the claims. Source indices come from the"
        " `claims` list."
    )
    experiments: Sequence[ConnectedEntity] = Field(
        description="Experiments designed to put methods in practice. Source indices"
        " come from the `methods` list."
    )

    @override
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""
        entities = [
            Entity(name=self.title, type=EntityType.TITLE),
            Entity(name=self.primary_area, type=EntityType.PRIMARY_AREA),
            *(Entity(name=kw, type=EntityType.KEYWORD) for kw in self.keywords),
            Entity(name=self.tldr, type=EntityType.TLDR),
            *(Entity(name=c.text, type=EntityType.CLAIM) for c in self.claims),
            *(Entity(name=m.text, type=EntityType.METHOD) for m in self.methods),
            *(
                Entity(name=x.text, type=EntityType.EXPERIMENT)
                for x in self.experiments
            ),
        ]

        relationships = [
            Relationship(source=self.title, target=self.primary_area),
            *(Relationship(source=self.title, target=kw) for kw in self.keywords),
            Relationship(source=self.title, target=self.tldr),
            *(Relationship(source=self.tldr, target=c.text) for c in self.claims),
            *(_relationships_from_indices(self.claims, self.methods, "claim->method")),
            *(
                _relationships_from_indices(
                    self.methods, self.experiments, "method->exp"
                )
            ),
        ]

        return Graph(
            title=title,
            abstract=abstract,
            entities=entities,
            relationships=relationships,
        )


class IndexedEntity(BaseModel):
    """Entity from the paper. It belongs to a list and carries its index in that list."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(description="Index of this entity in its original list.")
    text: str = Field(description="Sentence from the paper describing this entity.")


class ConnectedEntity(IndexedEntity):
    """Entity from a paper that has an index and is connected to other entities by index.

    The source indices from other entities are from the original list containing the
    connected entities.
    """

    source_indices: Sequence[int] = Field(
        description="Indices of the entities connected to this one in their original"
        " list."
    )


def _relationships_from_indices(
    sources: Sequence[IndexedEntity], targets: Iterable[ConnectedEntity], desc: str
) -> list[Relationship]:
    """For each target, find their source entities from `source_indices` by index.

    NB: If the target index is invald, it will be logged and the relationship will be
    skipped.

    Returns:
        List of all relationships between targets and their sources.
    """
    return [
        Relationship(source=source.text, target=target.text)
        for target in targets
        for source_idx in target.source_indices
        if (source := _at(sources, source_idx, desc))
    ]


def _at[T](seq: Sequence[T], idx: int, desc: str) -> T | None:
    """Get `seq[idx]` if possible, otherwise return None and log warning with `desc`."""
    try:
        return seq[idx]
    except IndexError:
        logger.warning("Invalid index at '%s': %d out of %d", desc, idx, len(seq))
        return None


class GPTGraphStrict2(GPTGraphBase):
    """Graph representing the paper."""

    # This is very similar to `GPTGraphStrict`, but there the connections are backwards.
    # E.g., `experiment` with a backlink to `method`. Here the connections are forward.
    # There are also dedicated classes for each entity, even at the bottom levels, with
    # different field names for each connection.

    title: str = Field(description="Title of the paper.")
    primary_area: str = Field(
        description="The primary subject area of the paper picked from the ICLR list of"
        " topics."
    )
    keywords: Sequence[str] = Field(
        description="Keywords that summarise the key aspects of the paper."
    )
    tldr: str = Field(description="Sentence that summarises the paper.")
    claims: Sequence[ClaimEntity] = Field(
        description="Main contributions the paper claims to make, with connections to"
        " target `methods`."
    )
    methods: Sequence[MethodEntity] = Field(
        description="Methods used to verify the claims, with connections to target"
        " `experiments`"
    )
    experiments: Sequence[ExperimentEntity] = Field(
        description="Experiments designed to put methods in practice."
    )

    @override
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""
        entities = [
            Entity(name=self.title, type=EntityType.TITLE),
            Entity(name=self.primary_area, type=EntityType.PRIMARY_AREA),
            *(Entity(name=kw, type=EntityType.KEYWORD) for kw in self.keywords),
            Entity(name=self.tldr, type=EntityType.TLDR),
            *(Entity(name=c.text, type=EntityType.CLAIM) for c in self.claims),
            *(Entity(name=m.text, type=EntityType.METHOD) for m in self.methods),
            *(
                Entity(name=x.text, type=EntityType.EXPERIMENT)
                for x in self.experiments
            ),
        ]

        relationships = [
            Relationship(source=self.title, target=self.primary_area),
            *(Relationship(source=self.title, target=kw) for kw in self.keywords),
            Relationship(source=self.title, target=self.tldr),
            *(Relationship(source=self.tldr, target=c.text) for c in self.claims),
            *(
                Relationship(source=c.text, target=target.text)
                for c in self.claims
                for midx in c.method_indices
                if (target := _at(self.methods, midx, "claim->method"))
            ),
            *(
                Relationship(source=m.text, target=target.text)
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

    text: str = Field(description="Description of a claim made by the paper")
    method_indices: Sequence[int] = Field(
        description="Indices for the `methods` connected to this claim in the `methods`"
        " list. There must be at least one connected `method`."
    )


class MethodEntity(BaseModel):
    """Entity representing a method described in the paper to support the claims."""

    text: str = Field(
        description="Description of a method used to validate claims from the paper."
    )
    index: int = Field(description="Index for this method in the `methods` list")
    experiment_indices: Sequence[int] = Field(
        description="Indices for the `experiments` connected to this method in the "
        " `experiments` list. There must be at least one connected `experiment`."
    )


class ExperimentEntity(BaseModel):
    """Entity representing an experiment used to validate a method from the paper."""

    text: str = Field(
        description="Description of an experiment used to validate the methods from"
        " the paper."
    )
    index: int = Field(description="Index for this method in the `experiments` list")


_GRAPH_TYPES: Mapping[str, type[GPTGraphBase]] = {
    "strict": GPTGraphStrict,
    "strict2": GPTGraphStrict2,
}


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit: int | None,
    graph_user_prompt: str,
    classify_user_prompt: str,
    output_dir: Path,
) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    logger.info(
        "CONFIG:\n"
        f"  Model: {model}\n"
        f"  Data path: {data_path.resolve()}\n"
        f"  Data hash (sha256): {data_hash}\n"
        f"  Output dir: {output_dir.resolve()}\n"
        f"  Limit: {limit if limit is not None else 'All'}\n"
        f"  Graph prompt: {graph_user_prompt}\n"
        f"  Classify prompt: {classify_user_prompt}\n"
    )


_GRAPH_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)
_PRIMARY_AREAS: list[str] = tomllib.loads(
    read_resource("gpt.prompts", "primary_areas.toml")
)["primary_areas"]
_GRAPH_USER_PROMPTS = load_prompts("extract_graph")


async def _generate_graph(
    client: AsyncOpenAI, example: Paper, model: str, user_prompt: PromptTemplate
) -> GPTResult[PromptResult[Graph]]:
    user_prompt_text = user_prompt.template.format(
        title=example.title,
        abstract=example.abstract,
        main_text=example.main_text(),
        primary_areas=", ".join(_PRIMARY_AREAS),
    )
    result = await run_gpt(
        _GRAPH_TYPES[user_prompt.type_name],
        client,
        _GRAPH_SYSTEM_PROMPT,
        user_prompt_text,
        model,
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
) -> GPTResult[list[PromptResult[Graph]]]:
    total_cost = 0
    graph_results: list[PromptResult[Graph]] = []

    tasks = [_generate_graph(client, example, model, user_prompt) for example in data]

    for task in as_completed(tasks, desc="Extracting graphs"):
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
        papers: Papers used to generate the graph
        graphs: Graphs generated from the paper. Must match the respective paper in
            `papers`
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

    for pg in paper_graphs:
        output.append(
            Output(
                paper=pg.paper.title,
                graphml=graph_to_digraph(pg.graph.item).graphml(),
                graph=pg.graph.item,
                prompt=pg.graph.prompt,
            )
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
        graph_user_prompt: Key to the prompt used to generate the Graph
        papers: Papers used to generate the graph
        graphs: Graphs generated from the paper. Must match the respective paper in
            `papers`
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


async def extract_graph(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit: int | None,
    graph_user_prompt_key: str,
    classify_user_prompt_key: str,
    display: bool,
    output_dir: Path,
    classify: bool,
    continue_papers_file: Path | None,
) -> None:
    """Extract graphs from the papers in the dataset and (maybe) classify them.

    The graphs follow a taxonomy based on paper title -> concepts -> sentences. The
    relationships between nodes are either supporting or contrasting.

    The papers should come from the ASAP-Review dataset as processed by the
    paper.asap module.

    The classification part is optional. It uses the generated graphs as input as saves
    the results (metrics and predictions) to {output_dir}/classification.

    Args:
        model: GPT model code. Must support Structured Outputs.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
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

    Returns:
        None. The output is saved to disk.
    """
    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if model not in MODELS_ALLOWED:
        raise ValueError(f"Model {model} not in allowed models: {MODELS_ALLOWED}")
    model = MODEL_SYNONYMS.get(model, model)

    _log_config(
        model=model,
        data_path=data_path,
        limit=limit,
        graph_user_prompt=graph_user_prompt_key,
        classify_user_prompt=classify_user_prompt_key,
        output_dir=output_dir,
    )

    client = AsyncOpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_bytes())

    papers = data[:limit]
    graph_user_prompt = _GRAPH_USER_PROMPTS[graph_user_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        Graph,
        output_intermediate_file,
        continue_papers_file,
        papers,
        continue_key=get_id,
        original_key=get_id,
    )
    if not papers_remaining.remaining:
        logger.info("No items left to process. They're all on the `continues` file.")
    else:
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
            client, model, paper_graphs, classify_user_prompt_key, output_dir
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


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        dest="subcommand",
        required=True,
    )

    # 'run' subcommand parser
    run_parser = subparsers.add_parser(
        "run",
        help="Run the main extraction process",
        description="Run the main extraction process with the provided arguments.",
    )

    # Add original arguments to the 'run' subcommand
    run_parser.add_argument(
        "data_path",
        type=Path,
        help="The path to the JSON file containing the papers data.",
    )
    run_parser.add_argument(
        "output_dir",
        type=Path,
        help="The path to the output directory where files will be saved.",
    )
    run_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        choices=MODELS_ALLOWED,
        help="The model to use for the extraction.",
    )
    run_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "The OpenAI API key to use for the extraction. Defaults to OPENAI_API_KEY"
            " env var. Can be read from the .env file."
        ),
    )
    run_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=1,
        help="The number of papers to process. Defaults to 1 example.",
    )
    run_parser.add_argument(
        "--graph-user-prompt",
        type=str,
        choices=_GRAPH_USER_PROMPTS.keys(),
        default="simple",
        help="The user prompt to use for the graph extraction. Defaults to %(default)s.",
    )
    run_parser.add_argument(
        "--classify-user-prompt",
        type=str,
        choices=_GRAPH_USER_PROMPTS.keys(),
        default="simple",
        help="The user prompt to use for paper classification. Defaults to %(default)s.",
    )
    run_parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the extracted graph.",
    )
    run_parser.add_argument(
        "--classify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Classify the papers based on the extracted entities.",
    )
    run_parser.add_argument(
        "--continue-papers",
        type=Path,
        default=None,
        help="Path to file with data from a previous run",
    )

    # 'prompts' subcommand parser
    prompts_parser = subparsers.add_parser(
        "prompts",
        help="List available prompts",
        description="List available prompts. Use --detail for more information.",
    )
    prompts_parser.add_argument(
        "--detail",
        action="store_true",
        help="Provide detailed descriptions of the prompts.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    setup_cli_parser(parser)

    args = parser.parse_args()
    setup_logging()

    if args.subcommand == "prompts":
        list_prompts(detail=args.detail)
    elif args.subcommand == "run":
        asyncio.run(
            extract_graph(
                args.model,
                args.api_key,
                args.data_path,
                args.limit,
                args.graph_user_prompt,
                args.classify_user_prompt,
                args.display,
                args.output_dir,
                args.classify,
                args.continue_papers,
            )
        )


def list_prompts(detail: bool) -> None:
    items = [
        ("GRAPH EXTRACTION PROMPTS", _GRAPH_USER_PROMPTS),
        ("CLASSIFICATION PROMPTS", CLASSIFY_USER_PROMPTS),
    ]
    for title, prompts in items:
        print()
        if detail:
            print(">>>", title)
        else:
            print(title)
        for key, prompt in prompts.items():
            if detail:
                sep = "-" * 80
                print(f"{sep}\n{key}\n{sep}\n{prompt}")
            else:
                print(f"- {key}")


if __name__ == "__main__":
    main()
