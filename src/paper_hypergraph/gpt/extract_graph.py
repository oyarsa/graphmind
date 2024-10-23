"""Generate graphs from papers from the ASAP-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper_hypergraph import hierarchical_graph
from paper_hypergraph.gpt.evaluate_graph import (
    CLASSIFY_USER_PROMPTS,
    evaluate_graphs,
)
from paper_hypergraph.gpt.model import Entity, EntityType, Graph, Paper, Relationship
from paper_hypergraph.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    Prompt,
    PromptResult,
    run_gpt,
)
from paper_hypergraph.util import Timer, setup_logging

logger = logging.getLogger("gpt.extract_graph")


class GPTRelationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_index: int
    target_index: int


class GPTEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    type: EntityType


class GPTGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: Sequence[GPTEntity]
    relationships: Sequence[GPTRelationship]

    def __str__(self) -> str:
        entities = "\n".join(
            f"  {e.index}. {e.type} - {e.name}"
            for e in sorted(self.entities, key=lambda e: e.index)
        )

        relationships = "\n".join(
            f" {i}. {r.source_index} - {r.target_index}"
            for i, r in enumerate(
                sorted(
                    self.relationships, key=lambda r: (r.source_index, r.target_index)
                ),
                1,
            )
        )
        node_type_counts = sorted(Counter(e.type for e in self.entities).items())

        return "\n".join(
            [
                f"Nodes: {len(self.entities)}",
                f"Edges: {len(self.relationships)}",
                f"Node types: {", ".join(f"{k}: {v}" for k, v in node_type_counts)}",
                "",
                "Entities:",
                entities,
                "",
                "Relationships:",
                relationships,
                "",
            ]
        )


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
_PRIMARY_AREAS = "unsupervised, self-supervised, semi-supervised, supervised representation learning, transfer learning, meta learning, lifelong learning, reinforcement learning, representation learning for computer vision, audio, language, other modalities, metric learning, kernel learning, sparse coding, probabilistic methods, Bayesian methods, variational inference, sampling, UQ, generative models, causal reasoning, optimization, learning theory, learning on graphs, other geometries, topologies, societal considerations, fairness, safety, privacy, visualization, interpretation of learned representations, datasets, benchmarks, infrastructure, software libraries, hardware, neurosymbolic, hybrid AI systems, physics-informed, logic, formal reasoning, applications to robotics, autonomy, planning, applications to neuroscience, cognitive science, applications to physical sciences, physics, chemistry, biology, general machine learning"

_GRAPH_USER_PROMPTS = {
    "introduction": f"""\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, the main text from the paper.

Your task is to extract entities of these types:
- title: the title of the paper
- primary area: what scientific primary area the paper is from. Pick one from \
{_PRIMARY_AREAS}.
- TLDR: a sentence that summarises the paper
- claim: summarise what the paper claims to contribute, especially claims made in the \
abstract, introduction, discussion and conclusion. Pay attention to the key phrases \
that highlight new findings or interpretations.
- method: for each claim, identify the methods used to validate the claims from the \
method sections. These include the key components: algorithms, theoretical framework \
or novel techniques introduced.
- experiment: what models, baselines, datasets, etc. that were used in \
experiments to validate the methods.

Extract these entities and the relationships between them as a graph. The paper title is \
the main node and represents the paper. There are restrictions for what types of \
connections can be made between node based on their types. The only allowed edges are:

- There can be no incoming edges to the paper title node
- Edges between nodes of the same type cannot exist
- The graph is hierarchical, and all edges are from a node type above to one below. The \
hierarchy is Title > Primary area = Keywords = TLDR > Claims > Methods > Experiments. \
Note that Title, Primary area and Keywords are on the same level.
- Title -> TLDR (1:1): there is only one TLDR node, and it's connected to the title
- Title -> primary area (1:1): there is only one primary area node, and it's connected \
to the title.
- Title -> keywords (1:N, N <= 5): there can be up to 5 keyword nodes, and they're \
connected to the title.
- TLDR -> claims (N:M): there can be many claim nodes, and they're connected to the TLDR \
node.
- Claims -> methods (N:M): there can be many methods nodes, and they're connected to the \
claim nodes. A claim can connect to multiple methods, and a method can connect to \
multiple claims.
- Methods -> experiments (N:M): there can be many experiments, and they're connected to \
the method nodes. An experiment can connect to multiple methods, and a method can \
connect to multiple claims.

Each entity must have a unique index. You must use these indexes to represent the \
relationships between the entities.

All entity types should be present in the output.

#####
-Data-
Title: {{title}}
Abstract: {{abstract}}

Main text:
{{main_text}}

#####
Output:
""",
    # TODO: Add new node types
    # Unfortunately, the models don't always comply with the rules, especially the
    # rule that each concept must connect to at least one supporting sentence. This
    # version is currently the best at that.
    "bullets": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, the main text from the paper.

Your task is to extract three types of entities and the relationships between them:
- title: the title of the paper
- concept: the top 5 key concepts mentioned in the abstract. If there are fewer than 5, \
use only those.
- sentences: sentences from the main text and especially the introduction that mention \
the key concepts.

Extract these entities and the relationships between them as a graph. The paper title is \
the only main node, connected to the key concepts. The key concepts are connected to the \
sentences that mention them.

You MUST follow these rules:

- There is only one main node (title) and it MUST be connected to all the key concepts.
- Only provide connections from title to concepts and concepts to sentences.
- Do NOT provide relationships between concepts to concepts or sentences to sentences.
- There can be multiple sentences for a single concept, and a single sentence can \
connect to multiple concepts.
- Each concept MUST connect to at least one sentence.
- Each sentence MUST connect to at least one concept.
- There MUST be twice as many sentences as concepts.
- There MUST be at least two sentences of each type.

All entities (title, concepts and sentences) MUST be mentioned in the output.

#####
-Data-
Title: {title}
Abstract: {abstract}

Main text:
{main_text}


#####
Output:
""",
}


def _generate_graphs(
    client: OpenAI, data: list[Paper], model: str, user_prompt: str
) -> GPTResult[list[PromptResult[Graph]]]:
    total_cost = 0
    graph_results: list[PromptResult[Graph]] = []

    for example in tqdm(data, desc="Extracting graphs"):
        prompt = user_prompt.format(
            title=example.title,
            abstract=example.abstract,
            main_text=example.main_text(),
        )
        result = run_gpt(GPTGraph, client, _GRAPH_SYSTEM_PROMPT, prompt, model)
        graph = (
            _graph_from_gpt_graph(result.result)
            if result.result
            else Graph(entities=[], relationships=[])
        )
        total_cost += result.cost
        graph_results.append(
            PromptResult(
                item=graph, prompt=Prompt(user=prompt, system=_GRAPH_SYSTEM_PROMPT)
            )
        )

    return GPTResult(graph_results, total_cost)


def _graph_from_gpt_graph(gpt_graph: GPTGraph) -> Graph:
    """Builds a graph from the GPT output.

    Assumes that the entities are all valid, and that the source/target indices for
    the relationships match the indices in the entities sequence.
    """
    entities = [Entity(name=e.name, type=e.type) for e in gpt_graph.entities]

    entity_index = {e.index: e for e in gpt_graph.entities}
    relationships = [
        Relationship(
            source=entity_index[r.source_index].name,
            target=entity_index[r.target_index].name,
        )
        for r in gpt_graph.relationships
    ]

    return Graph(entities=entities, relationships=relationships)


def _save_graphs(
    papers: Iterable[Paper],
    graph_results: Iterable[PromptResult[Graph]],
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
        graph: str
        prompt: Prompt

    output: list[Output] = []

    for paper, graph_result in zip(papers, graph_results):
        output.append(
            Output(
                paper=paper.title,
                graph=_graph_to_dag(graph_result.item).graphml(),
                prompt=graph_result.prompt,
            )
        )

    (output_dir / "result_graphs.json").write_bytes(
        TypeAdapter(list[Output]).dump_json(output, indent=2)
    )


def _display_graphs(
    model: str,
    graph_user_prompt_key: str,
    papers: Iterable[Paper],
    graph_results: Iterable[PromptResult[Graph]],
    output_dir: Path,
    visualise: bool,
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
        visualise: If True, show the graph on screen. This suspends the process until
            the plot is closed.
    """
    for paper, graph_result in zip(papers, graph_results):
        dag = _graph_to_dag(graph_result.item)

        try:
            dag.visualise_hierarchy(
                img_path=output_dir / f"{paper.title}.png",
                display_gui=visualise,
                description=f"index - model: {model} - prompt: {graph_user_prompt_key}",
            )
        except hierarchical_graph.GraphError:
            logger.exception("Error visualising graph")


def extract_graph(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit: int | None,
    graph_user_prompt_key: str,
    classify_user_prompt_key: str,
    visualise: bool,
    output_dir: Path,
    classify: bool,
) -> None:
    """Extract graphs from the papers in the dataset and (maybe) classify them.

    The graphs follow a taxonomy based on paper title -> concepts -> sentences. The
    relationships between nodes are either supporting or contrasting.

    The papers should come from the ASAP-Review dataset as processed by the
    paper_hypergraph.asap module.

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
        visualise: If True, show each graph on screen. This suspends the process until
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

    client = OpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_bytes())

    papers = data[:limit]
    graph_user_prompt = _GRAPH_USER_PROMPTS[graph_user_prompt_key]

    with Timer() as timer_gen:
        graph_results = _generate_graphs(client, papers, model, graph_user_prompt)

    logger.info(f"Graph generation time elapsed: {timer_gen.human}")
    logger.info(f"Total graph generation cost: ${graph_results.cost:.10f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_graphs(papers, graph_results.result, output_dir)
    _display_graphs(
        model,
        graph_user_prompt_key,
        papers,
        graph_results.result,
        output_dir,
        visualise,
    )

    if classify:
        graphs = [result.item for result in graph_results.result]
        evaluate_graphs(
            client, model, papers, graphs, classify_user_prompt_key, output_dir
        )


def _graph_to_dag(graph: Graph) -> hierarchical_graph.DiGraph:
    return hierarchical_graph.DiGraph.from_elements(
        nodes=[hierarchical_graph.Node(e.name, e.type.value) for e in graph.entities],
        edges=[
            hierarchical_graph.Edge(r.source, r.target) for r in graph.relationships
        ],
    )


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
        default="bullets",
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
        "--visualise",
        "-V",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visualise the extracted graph.",
    )
    run_parser.add_argument(
        "--classify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Classify the papers based on the extracted entities.",
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
    setup_logging("gpt")

    if args.subcommand == "prompts":
        list_prompts(detail=args.detail)
    elif args.subcommand == "run":
        extract_graph(
            args.model,
            args.api_key,
            args.data_path,
            args.limit,
            args.graph_user_prompt,
            args.classify_user_prompt,
            args.visualise,
            args.output_dir,
            args.classify,
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
