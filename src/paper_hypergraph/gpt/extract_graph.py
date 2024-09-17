"""Extract the entities graph from a text using GPT-4."""

import argparse
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    introduction: str


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

def run_gpt_structured[T](
    output_type: type[T],
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> T:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=output_type,
        seed=0,
        temperature=0,
    )

    parsed = completion.choices[0].message.parsed
    if not parsed:
        return output_type()
    return parsed
_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)


def extract_graph(model: str, api_key: str | None, data_path: Path) -> None:
    if not api_key:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "The OPENAI_API_KEY must be provided as env var or argument."
            )
        api_key = os.environ["OPENAI_API_KEY"]

    model = _MODEL_SYNONYMS.get(model, model)

    client = OpenAI(api_key=api_key)

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_text())
    print(len(data))

    example = """
    Alice and Bob are friends. Bob and Charlie are roommates. Charlile and Bob are also
    coworkers.
    """
    graph = run_gpt_structured(Graph, client, SYSTEM_PROMPT, example, model)

    print(graph)


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
        type=str,
        default="gpt-4o-2024-08-06",
        help="The model to use for the extraction.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The OpenAI API key to use for the extraction.",
    )

    args = parser.parse_args()
    extract_graph(args.model, args.api_key, args.data_path)


if __name__ == "__main__":
    main()
