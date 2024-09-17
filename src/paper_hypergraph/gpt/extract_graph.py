"""Extract the entities graph from a text using GPT-4."""

import argparse
import hashlib
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter


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
    description: str


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: str


class Graph(BaseModel):
    model_config = ConfigDict(frozen=True)

    concepts: Sequence[Entity]
    relationships: Sequence[Relationship]

    def __str__(self) -> str:
        entities = "\n".join(
            f"  {i}. {c.name} - {c.type}" for i, c in enumerate(self.concepts, 1)
        )

        relationships = "\n".join(
            f" {i}. {r.source} - {r.description} - {r.target}"
            for i, r in enumerate(
                sorted(
                    self.relationships,
                    key=lambda r: (r.source, r.target, r.description),
                ),
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


# Cost in $ per 1M tokens
# From https://openai.com/api/pricing/
_MODEL_COSTS = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
}


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_cost, output_cost = _MODEL_COSTS[model]
    return input_cost / 1e6 * input_tokens + output_cost / 1e6 * output_tokens


@dataclass(frozen=True)
class ModelResult:
    graph: Graph
    cost: float


def run_gpt_graph(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> ModelResult:
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

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed
    if not parsed:
        graph = Graph(concepts=[], relationships=[])
    else:
        graph = parsed

    return ModelResult(graph=graph, cost=cost)


def _log_config(*, model: str, data_path: Path, limit: int | None) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    print("CONFIG:")
    print(f"  Model: {model}")
    print(f"  Data path: {data_path.resolve()}")
    print(f"  Data hash: {data_hash}")
    print(f"  Limit: {limit if limit is not None else 'All'}")
    print()


_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)

_USER_PROMPT = """\
The following text contains information about a scientific paper. It includes the \
paper's title and abstract.

Your task is to extract the the top 5 key concepts mentioned in the abstract and the \
relationships between them. Do not provide relatinshiops between concepts beyond the \
top 5. If there are fewer than 5 concepts, use only those.

#####
-Data-
Title: {title}
Abstract: {abstract}
#####
Output:
"""


def run_data(client: OpenAI, data: list[Paper], model: str) -> None:
    total_cost = 0
    for example in data:
        prompt = _USER_PROMPT.format(title=example.title, abstract=example.abstract)
        result = run_gpt_graph(client, _SYSTEM_PROMPT, prompt, model)
        total_cost += result.cost
        print("Example:")
        print(example)
        print()
        print("Graph:")
        print(result.graph)
        print()

    print(f"\n\nTotal cost: ${total_cost:.10f}")


def extract_graph(
    model: str, api_key: str | None, data_path: Path, limit: int | None
) -> None:
    if not api_key:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "The OPENAI_API_KEY must be provided as env var or argument."
            )
        api_key = os.environ["OPENAI_API_KEY"]

    model = _MODEL_SYNONYMS.get(model, model)

    _log_config(model=model, data_path=data_path, limit=limit)

    client = OpenAI(api_key=api_key)

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_text())

    time_start = time.perf_counter()
    run_data(client, data[:limit], model)
    time_elapsed = time.perf_counter() - time_start
    print(f"Time elapsed: {_convert_time_elapsed(time_elapsed)}")


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

    args = parser.parse_args()
    extract_graph(args.model, args.api_key, args.data_path, args.limit)


if __name__ == "__main__":
    main()
