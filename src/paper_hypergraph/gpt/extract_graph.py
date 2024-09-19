"""Extract the entities graph from a text using GPT-4."""

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import colorlog
import dotenv
import networkx as nx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper_hypergraph.graph import (
    GraphError,
    save_graph,
    validate_hierarchy_graph,
    visualise_hierarchy,
)

logger = logging.getLogger("extract_graph")


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    introduction: str

    def __str__(self) -> str:
        return f"Title: {self.title}\nAbstract: {self.abstract}\n"


class GptRelationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_index: int
    target_index: int


class EntityType(StrEnum):
    TITLE = "title"
    CONCEPT = "concept"
    SUPPORT = "support"


class GptEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    type: EntityType


class GptGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: Sequence[GptEntity]
    relationships: Sequence[GptRelationship]

    @classmethod
    def empty(cls) -> Self:
        return cls(entities=[], relationships=[])

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

        return "\n".join(
            [
                f"Nodes: {len(self.entities)}",
                f"Edges: {len(self.relationships)}",
                f"Titles: {sum(e.type == EntityType.TITLE for e in self.entities)}",
                f"Concepts: {sum(e.type == EntityType.CONCEPT for e in self.entities)}",
                f"Supports: {sum(e.type == EntityType.SUPPORT for e in self.entities)}",
                "",
                "Entities:",
                entities,
                "",
                "Relationships:",
                relationships,
                "",
            ]
        )


class Relationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: EntityType


class Graph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: Sequence[Entity]
    relationships: Sequence[Relationship]

    @classmethod
    def from_gpt_graph(cls, gpt_graph: GptGraph) -> Self:
        entities = [Entity(name=e.name, type=e.type) for e in gpt_graph.entities]

        entity_index = {e.index: e for e in gpt_graph.entities}
        relationships = [
            Relationship(
                source=entity_index[r.source_index].name,
                target=entity_index[r.target_index].name,
            )
            for r in gpt_graph.relationships
        ]

        return cls(entities=entities, relationships=relationships)

    def __str__(self) -> str:
        entities = "\n".join(
            f"  {i}. {c.type} - {c.name}"
            for i, c in enumerate(
                sorted(self.entities, key=lambda e: (e.type, e.name)), 1
            )
        )

        relationships = "\n".join(
            f" {i}. {r.source} - {r.target}"
            for i, r in enumerate(
                sorted(self.relationships, key=lambda r: (r.source, r.target)),
                1,
            )
        )

        return "\n".join(
            [
                f"Nodes: {len(self.entities)}",
                f"Edges: {len(self.relationships)}",
                f"Titles: {sum(e.type == EntityType.TITLE for e in self.entities)}",
                f"Concepts: {sum(e.type == EntityType.CONCEPT for e in self.entities)}",
                f"Supports: {sum(e.type == EntityType.SUPPORT for e in self.entities)}",
                "",
                "Entities:",
                entities,
                "",
                "Relationships:",
                relationships,
                "",
            ]
        )


_MODEL_SYNONYMS = {
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
    "gpt-4o": "gpt-4o-2024-08-06",
}


# Cost in $ per 1M tokens: (input cost, output cost)
# From https://openai.com/api/pricing/
_MODEL_COSTS = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_cost, output_cost = _MODEL_COSTS[model]
    return prompt_tokens / 1e6 * input_cost + completion_tokens / 1e6 * output_cost


@dataclass(frozen=True)
class ModelResult:
    graph: GptGraph
    cost: float


def run_gpt_graph(
    client: OpenAI, system_prompt: str, user_prompt: str, model: str
) -> ModelResult:
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=GptGraph,
            seed=0,
            temperature=0,
        )
    except Exception:
        logger.exception("Error making API request")
        return ModelResult(graph=GptGraph.empty(), cost=float("nan"))

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed
    if not parsed:
        graph = GptGraph.empty()
    else:
        graph = parsed

    return ModelResult(graph=graph, cost=cost)


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit: int | None,
    user_prompt: str,
    output_dir: Path,
) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    logger.info(
        "CONFIG:"
        f"  Model: {model}\n"
        f"  Data path: {data_path.resolve()}\n"
        f"  Data hash: {data_hash}\n"
        f"  Output dir: {output_dir.resolve()}\n"
        f"  Limit: {limit if limit is not None else 'All'}\n"
        f"  User prompt: {user_prompt}\n"
    )


_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)

_USER_PROMPTS = {
    "introduction": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and introduction.

Your task is to extract three types of entities:
- title: the title of the paper
- concepts: the top 5 key concepts mentioned in the abstract. If there are fewer than 5, \
use only those.
- supports: supporting sentences from the introduction that mention the key concepts.

Extract these entities and the relationships between them as a graph. The paper title is \
the main node, connected to the key concepts. The key concepts are connected to the \
supporting sentences that mention them.

Each entity must have a unique index. You must use these indexes to represent the \
relationships between the entities.

Only provide connections between the entities from each of the three types (title to \
concepts, concepts to supporting sentences). Do not provide relationships among concepts \
or supporting sentences.

The supporting sentences count as entities and must be returned along with the title \
and the concepts. There can be multiple supporting sentences for a single concept, and a single \
support sentence can connect to multiple concepts. There can be up to 10 supporting sentences. \
Each concept must be connected to at least one supporting sentence, and each supporting sentence must be \
connected to at least one concept.

All entities (title, concepts and supports) should be mentioned in the output.

#####
-Data-
Title: {title}
Abstract: {abstract}

Introduction:
{introduction}

#####
Output:
""",
    # Unfortunately, the models don't always comply with the rules, especially the
    # rule that each concept must connect to at least one supporting sentence. This
    # version is currently the best at that.
    "bullets": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and introduction.

Your task is to extract three types of entities and the relationships between them:
- title: the title of the paper
- concept: the top 5 key concepts mentioned in the abstract. If there are fewer than 5, \
use only those.
- support: supporting sentences from the introduction that mention the key concepts.

Extract these entities and the relationships between them as a graph. The paper title is \
the only main node, connected to the key concepts. The key concepts are connected to the \
supporting sentences that mention them.

You MUST follow these rules:

- There is only one main node (title) and it MUST be connected to all the key concepts.
- Only provide connections from title to concepts and concepts to supporting sentences.
- Do NOT provide relationships between concepts or supporting sentences.
- There can be multiple supporting sentences for a single concept, and a single \
support sentence can connect to multiple concepts.
- Each concept MUST connect to at least one supporting sentence.
- Each supporting sentence MUST connect to at least one concept.
- There MUST be twice as many supporting sentences as concepts.

All entities (title, concepts and supports) MUST be mentioned in the output.

#####
-Data-
Title: {title}
Abstract: {abstract}
Introduction:
{introduction}

#####
Output:
""",
}


def run_data(
    client: OpenAI, data: list[Paper], model: str, user_prompt: str
) -> list[Graph]:
    total_cost = 0
    graphs: list[Graph] = []

    for example in tqdm(data):
        prompt = user_prompt.format(
            title=example.title,
            abstract=example.abstract,
            introduction=example.introduction,
        )
        result = run_gpt_graph(client, _SYSTEM_PROMPT, prompt, model)
        graph = Graph.from_gpt_graph(result.graph)
        total_cost += result.cost

        supports = (sum(e.type == EntityType.SUPPORT for e in graph.entities),)
        valid = validate_hierarchy_graph(graph_to_networkx_dag(graph))

        logger.debug(
            "Example:\n"
            f"{example}\n\n"
            "Graph:\n"
            f"{graph}\n\n"
            f"Number of supports: {supports}\n"
            f"Graph validation: {valid or 'Valid'}\n"
        )

        graphs.append(graph)

    logger.info(f"Total cost: ${total_cost:.10f}")

    return graphs


def extract_graph(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit: int | None,
    user_prompt_key: str,
    visualise: bool,
    output_dir: Path,
) -> None:
    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = _MODEL_SYNONYMS.get(model, model)

    _log_config(
        model=model,
        data_path=data_path,
        limit=limit,
        user_prompt=user_prompt_key,
        output_dir=output_dir,
    )

    client = OpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_text())
    user_prompt = _USER_PROMPTS[user_prompt_key]

    time_start = time.perf_counter()

    papers = data[:limit]
    graphs = run_data(client, papers, model, user_prompt)

    time_elapsed = time.perf_counter() - time_start
    logger.info(f"Time elapsed: {_convert_time_elapsed(time_elapsed)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for paper, graph in zip(papers, graphs):
        dag = graph_to_networkx_dag(graph)
        save_graph(dag, output_dir / f"{paper.title}.graphml")

        try:
            visualise_hierarchy(
                dag,
                show=visualise,
                img_path=output_dir / f"{paper.title}.png",
                description=f"index - model: {model} - prompt: {user_prompt_key}",
            )
        except GraphError:
            logger.exception("Error visualising graph")


def graph_to_networkx_dag(graph: Graph) -> nx.DiGraph:
    g = nx.DiGraph()

    for entity in graph.entities:
        g.add_node(entity.name, type=entity.type.value)

    for relationship in graph.relationships:
        g.add_edge(relationship.source, relationship.target)

    return g


def _convert_time_elapsed(seconds: float) -> str:
    """Convert a time duration from seconds to a human-readable format."""
    units = [("d", 86400), ("h", 3600), ("m", 60)]
    parts: list[str] = []

    for name, count in units:
        value, seconds = divmod(seconds, count)
        if value >= 1:
            parts.append(f"{int(value)}{name}")

    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f}s")

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="The path to the JSON file containing the papers data.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The path to the output directory where files will be saved.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        help="The model to use for the extraction.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The OpenAI API key to use for the extraction. Defaults to OPENAI_API_KEY"
        " env var. Can be read from the .env file.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=1,
        help="The number of papers to process. Defaults to 1 example.",
    )
    parser.add_argument(
        "--user-prompt",
        "-P",
        type=str,
        choices=_USER_PROMPTS.keys(),
        default="bullets",
        help="The user prompt to use for the extraction. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--visualise",
        "-V",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visualise the extracted graph.",
    )
    args = parser.parse_args()
    setup_logging(logger)

    extract_graph(
        args.model,
        args.api_key,
        args.data_path,
        args.limit,
        args.user_prompt,
        args.visualise,
        args.output_dir,
    )


def setup_logging(logger: logging.Logger) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = colorlog.StreamHandler()

    fmt = "%(log_color)s%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(colorlog.ColoredFormatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(handler)


if __name__ == "__main__":
    main()
