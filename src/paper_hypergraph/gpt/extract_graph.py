"""Extract the entities graph from a text using GPT-4."""

import argparse
import hashlib
import logging
import os
import textwrap
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import colorlog
import dotenv
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter

logger = logging.getLogger("extract_graph")


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    introduction: str

    def __str__(self) -> str:
        return f"Title: {self.title}\nAbstract: {self.abstract}\n"


class Relationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str


class Graph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: Sequence[Entity]
    relationships: Sequence[Relationship]

    @classmethod
    def empty(cls) -> Self:
        return cls(entities=[], relationships=[])

    def __str__(self) -> str:
        entities = "\n".join(
            f"  {i}. {c.name} - {c.type}" for i, c in enumerate(self.entities, 1)
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
    "gpt-4o": "gpt-4o-mini-2024-07-18",
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
    graph: Graph
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
            response_format=Graph,
            seed=0,
            temperature=0,
        )
    except Exception:
        logger.exception("Error making API request")
        return ModelResult(graph=Graph.empty(), cost=float("nan"))

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed
    if not parsed:
        graph = Graph.empty()
    else:
        graph = parsed

    return ModelResult(graph=graph, cost=cost)


def _log_config(
    *, model: str, data_path: Path, limit: int | None, user_prompt: str
) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    out: list[str] = []

    out.append("CONFIG:")
    out.append(f"  Model: {model}")
    out.append(f"  Data path: {data_path.resolve()}")
    out.append(f"  Data hash: {data_hash}")
    out.append(f"  Limit: {limit if limit is not None else 'All'}")
    out.append(f"  User prompt: {user_prompt}")
    out.append("")

    logger.info("\n".join(out))


_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)

_USER_PROMPTS = {
    "abstract_only": """\
The following text contains information about a scientific paper. It includes the \
paper's title and abstract.

Your task is to extract the top 5 key concepts mentioned in the abstract and the \
relationships between them. Do not provide relatinshiops between concepts beyond the \
top 5. If there are fewer than 5 concepts, use only those.

#####
-Data-
Title: {title}
Abstract: {abstract}
#####
Output:
""",
    "introduction": """\
The following text contains information about a scientific paper. It includes the \
paper's title, abstract, and introduction.

Your task is to extract three types of entities:
- title: the title of the paper
- concepts: the top 5 key concepts mentioned in the abstract. If there are fewer than 5, \
use only those.
- supports: sentences in the introduction that mention the key concepts.

Then extract the relationships between these entities. The paper title is the main node, \
connected to the key concepts. The key concepts are connected to the supporting sentences
that mention them.

Only provide sentences between the entities from the three types (title to concepts, \
concetps to supports). Do not provide relationships between concepts or supports.

The supporting sentences count as entities and should be return along with the title and
the concepts.

All entities (title, concepts and supports) should be mentioned in the output.

#####
-Data-
Title: {title}
Abstract: {abstract}
#####
Output:
""",
}


def run_data(
    client: OpenAI, data: list[Paper], model: str, user_prompt: str
) -> list[Graph]:
    total_cost = 0
    graphs: list[Graph] = []

    for example in data:
        prompt = user_prompt.format(title=example.title, abstract=example.abstract)
        result = run_gpt_graph(client, _SYSTEM_PROMPT, prompt, model)
        total_cost += result.cost

        out: list[str] = []
        out.append("Example:")
        out.append(str(example))
        out.append("Graph:")
        out.append(str(result.graph))
        logger.debug("\n".join(out))

        graphs.append(result.graph)

    logger.info(f"Total cost: ${total_cost:.10f}")

    return graphs


def extract_graph(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit: int | None,
    user_prompt_key: str,
    visualise: bool,
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
    )

    client = OpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_text())
    user_prompt = _USER_PROMPTS[user_prompt_key]

    time_start = time.perf_counter()

    graphs = run_data(client, data[:limit], model, user_prompt)

    time_elapsed = time.perf_counter() - time_start
    logger.info(f"Time elapsed: {_convert_time_elapsed(time_elapsed)}")

    if visualise:
        for graph in graphs:
            visualise_tree(graph_to_networkx_tree(graph))


def graph_to_networkx_tree(graph: Graph) -> nx.DiGraph:
    g = nx.DiGraph()

    for entity in graph.entities:
        g.add_node(entity.name, type=entity.type)

    for relationship in graph.relationships:
        g.add_edge(relationship.source, relationship.target)

    return g


def visualise_tree(g: nx.DiGraph) -> None:
    # Identify root nodes (nodes with in-degree 0)
    in_degrees = cast(nx.classes.reportviews.DiDegreeView, g.in_degree())
    roots = [node for node, in_degree in in_degrees if in_degree == 0]

    if not roots:
        raise ValueError(
            "The graph doesn't have any root nodes (nodes with in-degree 0)"
        )

    # Compute the depth of each node
    def node_depth(node: str) -> int:
        if g.in_degree(node) == 0:
            return 0
        return 1 + max(node_depth(parent) for parent in g.predecessors(node))

    depths: dict[str, int] = {node: node_depth(node) for node in g.nodes()}
    max_depth = max(depths.values())

    # Create Hierarchical position mapping
    pos: dict[str, tuple[float, float]] = {}
    nodes_at_depth: dict[int, list[str]] = {d: [] for d in range(max_depth + 1)}

    for node, depth in depths.items():
        nodes_at_depth[depth].append(node)

    for depth, nodes in nodes_at_depth.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - (width - 1) / 2) / max(width - 1, 1), -depth)

    plt.figure(figsize=(20, 12))

    # Draw edges with arrows
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes and labels with wrapped text
    for node, (x, y) in pos.items():
        node_type = g.nodes[node].get("type", "")

        # Wrap the text to fit in the box
        wrapped_text = textwrap.wrap(node, width=20)

        # Calculate box dimensions
        box_width = 0.15
        # Adjust height based on number of lines
        box_height = 0.05 * (len(wrapped_text) + 1)

        rect = Rectangle(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            fill=True,
            facecolor="lightblue",
            edgecolor="black",
        )
        plt.gca().add_patch(rect)

        # Add wrapped text
        plt.text(
            x,
            y,
            "\n".join(wrapped_text),
            ha="center",
            va="center",
            wrap=True,
            fontsize=8,
        )

        # Add node type
        plt.text(
            x,
            y + box_height / 2 + 0.02,
            node_type,
            ha="center",
            va="bottom",
            color="red",
            fontsize=8,
        )

    plt.title("Paper Hierarchical Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


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
        "--model",
        "-m",
        type=str,
        default="gpt-4o-2024-08-06",
        help="The model to use for the extraction.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The OpenAI API key to use for the extraction. Defaults to OPENAI_API_KEY"
        " env var.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="The number of papers to process. Defaults to all.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        choices=_USER_PROMPTS.keys(),
        default="abstract_only",
        help="The user prompt to use for the extraction. Defaults to 'abstract_only'.",
    )
    parser.add_argument(
        "--visualise",
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
