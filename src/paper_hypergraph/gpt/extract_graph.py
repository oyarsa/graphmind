"""Extract the entities graph from a text using GPT-4."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import colorlog
import dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper_hypergraph import evaluation_metrics, hierarchical_graph

logger = logging.getLogger("extract_graph")

RATING_APPROVAL_THRESHOLD = 5
"""A rating is an approval if it's greater of equal than this."""


class RelationType(StrEnum):
    SUPPORT = "support"
    CONTRAST = "contrast"


class GptRelationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_index: int
    target_index: int
    type: RelationType


class EntityType(StrEnum):
    TITLE = "title"
    CONCEPT = "concept"
    SENTENCE = "sentence"


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
                f"Sentences: {sum(e.type == EntityType.SENTENCE for e in self.entities)}",
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
    type: RelationType


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
                type=r.type,
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
            f" {i}. {r.source} - {r.type} - {r.target}"
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
                f"Sentences: {sum(e.type == EntityType.SENTENCE for e in self.entities)}",
                "",
                "Entities:",
                entities,
                "",
                "Relationships:",
                relationships,
                "",
            ]
        )


def validate_rules(graph: Graph) -> str | None:
    """Check if graph rules hold. Returns error message if invalid, or None if valid.

    Rules:
    1. There must be exactly one Title node
    2. The Title node cannot have incoming edges
    3. The Title node can only have outgoing edges to Concepts, and these edges must
       be of type Support
    4. Concepts must have exactly one incoming edge each, and it must be the Title
    5. All outgoing edges from Concepts must be Sentences
    6. Sentences must not have outgoing edges

    Note: this doesn't throw an exception if the graph is invalid, it just returns
    the error message. The graph is allowed to be invalid, but it's useful to know
    why it's invalid.

    Returns:
        Error message describing the rule violated if the graph is invalid.
        None if the graph is follows all rules.
    """
    entities = {entity.name: entity for entity in graph.entities}
    incoming: defaultdict[str, list[Relationship]] = defaultdict(list)
    outgoing: defaultdict[str, list[Relationship]] = defaultdict(list)

    for relation in graph.relationships:
        incoming[relation.target].append(relation)
        outgoing[relation.source].append(relation)

    # Rule 1: Exactly one Title node
    titles = [e for e in graph.entities if e.type is EntityType.TITLE]
    if len(titles) != 1:
        return f"Found {len(titles)} title nodes. Should be exactly 1."

    title = titles[0]

    # Rule 2: Title node cannot have incoming edges
    if incoming[title.name]:
        return "Title node should not have any incoming edges."

    # Rule 3: Title's outgoing edges only to Concepts with Support type
    if any(
        entities[r.target].type is not EntityType.CONCEPT
        or r.type is not RelationType.SUPPORT
        for r in outgoing[title.name]
    ):
        return "Title should only have outgoing Support edges to Concepts."

    # Rule 4: Concepts must have exactly one incoming edge from Title
    # Rule 5: Concepts' outgoing edges must only link to Sentences
    for concept in (e for e in graph.entities if e.type is EntityType.CONCEPT):
        concept_incoming = incoming[concept.name]
        if (
            len(concept_incoming) != 1
            or entities[concept_incoming[0].source].type is not EntityType.TITLE
        ):
            return (
                f"Concept {concept.name} must have exactly one"
                " incoming edge from Title."
            )

        if any(
            entities[r.target].type is not EntityType.SENTENCE
            for r in outgoing[concept.name]
        ):
            return (
                f"Concept {concept.name} must only have outgoing" " edges to Sentences."
            )

    # Rule 6: Sentences must not have outgoing edges
    sentences = [e.name for e in graph.entities if e.type is EntityType.SENTENCE]
    if any(outgoing[s] for s in sentences):
        return "Sentences must not have outgoing edges."

    return None


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


_CLASSIFY_SYSTEM_PROMPT = (
    "Approve or reject the scientific paper based on the extracted entities."
)
_CLASSIFY_USER_PROMPTS = {
    "simple": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and a graph representation of the paper and its network.

Based on the extracted entities graph, approve or reject the paper. First, generate the \
rationale for your decision, then give the final decision.
"""
}


@dataclass(frozen=True)
class GptResult[T]:
    result: T
    cost: float


def run_gpt_graph(
    client: OpenAI, system_prompt: str, user_prompt: str, model: str
) -> GptResult[GptGraph]:
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
        return GptResult(result=GptGraph.empty(), cost=float("nan"))

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed
    result = parsed if parsed else GptGraph.empty()

    return GptResult(result=result, cost=cost)


class GptClassify(BaseModel):
    model_config = ConfigDict(frozen=True)

    rationale: str
    approved: bool

    @classmethod
    def empty(cls) -> Self:
        return cls(rationale="", approved=False)


def run_gpt_classify(
    client: OpenAI, system_prompt: str, user_prompt: str, model: str
) -> GptResult[GptClassify]:
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=GptClassify,
            seed=0,
            temperature=0,
        )
    except Exception:
        logger.exception("Error making API request")
        return GptResult(result=GptClassify.empty(), cost=float("nan"))

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed
    result = parsed if parsed else GptClassify.empty()

    return GptResult(result=result, cost=cost)


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
        "CONFIG:\n"
        f"  Model: {model}\n"
        f"  Data path: {data_path.resolve()}\n"
        f"  Data hash (sha256): {data_hash}\n"
        f"  Output dir: {output_dir.resolve()}\n"
        f"  Limit: {limit if limit is not None else 'All'}\n"
        f"  User prompt: {user_prompt}\n"
    )


_GRAPH_SYSTEM_PROMPT = (
    "Extract the entities from the text and the relationships between them."
)

_GRAPH_USER_PROMPTS = {
    "introduction": """\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and introduction.

Your task is to extract three types of entities:
- title: the title of the paper
- concepts: the top 5 key concepts mentioned in the abstract. If there are fewer than 5, \
use only those.
- sentences: sentences from the introduction that mention the key concepts.

Extract these entities and the relationships between them as a graph. The paper title is \
the main node, connected to the key concepts. The key concepts are connected to the \
sentences that mention them.

Each entity must have a unique index. You must use these indexes to represent the \
relationships between the entities.

Only provide connections between the entities from each of the three types (title to \
concepts, concepts to sentences). Do not provide relationships among concepts \
or sentences.

The sentences count as entities and must be returned along with the title \
and the concepts. There can be multiple sentences for a single concept, and a single \
sentence can connect to multiple concepts. There can be up to 10 sentences. \
Each concept must be connected to at least one sentence, and each sentence must be \
connected to at least one concept.

Relations can be of two types: supporting or contrasting. Supporting relations \
support the key concepts, provide evidence of why they might be true, or explain \
them. For example, they can be supporting sentences from citations, explanations from \
methodology or discussion sections. Constrasting relations oppose the main \
concepts. For example, they can be limitations from other works, negative results, or \
citations to other papers that did something differently.

Note that the relation between title and concepts is always supporting.

All entities (title, concepts and sentences) should be mentioned in the output.

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
- sentences: sentences from the introduction that mention the key concepts.

Extract these entities and the relationships between them as a graph. The paper title is \
the only main node, connected to the key concepts. The key concepts are connected to the \
sentences that mention them.

You MUST follow these rules:

- There is only one main node (title) and it MUST be connected to all the key concepts.
- Only provide connections from title to concepts and concepts to sentences.
- Do NOT provide relationships between concepts to concepts or sentences to sentences.
- There can be multiple sentences for a single concept, and a single \
sentence can connect to multiple concepts.
- Each concept MUST connect to at least one sentence.
- Each sentence MUST connect to at least one concept.
- There MUST be twice as many sentences as concepts.
- Relations can be of two types: supporting or contrasting.
- Supporting relations support the key concepts, provide evidence of why they might be \
  true, or explain them. For example, they can be supporting sentences from citations, \
  explanations from methodology or discussion sections. They are used to convince the \
  reader that the key concepts are valid.
- Constrasting relations oppose the main concepts. For example, they can be descriptions \
  of limitations from other works, negative results, or citations to other papers that \
  did something differently. They are used to convince the reader that the authors are \
  aware of different perspectives and have considered them.

All entities (title, concepts and sentences) MUST be mentioned in the output.

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


class RatingEvaluationStrategy(StrEnum):
    MEAN = "mean"
    """Mean rating is higher than the threshold."""
    MAJORITY = "majority"
    """Majority of ratings are higher than the threshold."""
    DEFAULT = MEAN

    def is_approved(self, ratings: Sequence[int]) -> bool:
        match self:
            case RatingEvaluationStrategy.MEAN:
                mean = sum(ratings) / len(ratings)
                return mean >= RATING_APPROVAL_THRESHOLD
            case RatingEvaluationStrategy.MAJORITY:
                approvals = [r >= RATING_APPROVAL_THRESHOLD for r in ratings]
                return sum(approvals) >= len(approvals) / 2


class PaperSection(BaseModel):
    heading: str
    text: str


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    introduction: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]

    def is_approved(
        self, evaluation: RatingEvaluationStrategy = RatingEvaluationStrategy.DEFAULT
    ) -> bool:
        return evaluation.is_approved(self.ratings)

    def __str__(self) -> str:
        return (
            f"Title: {self.title}\nAbstract: {self.abstract}\nRatings: {self.ratings}\n"
        )


class PaperResult(Paper):
    y_true: bool
    y_pred: bool


def _calculate_metrics(papers: Sequence[PaperResult]) -> evaluation_metrics.Metrics:
    return evaluation_metrics.calculate_metrics(
        [p.y_true for p in papers], [p.y_pred for p in papers]
    )


def _classify_papers(
    client: OpenAI,
    model: str,
    user_prompt_template: str,
    papers: Sequence[Paper],
    graphs: Sequence[Graph],
    output_dir: Path,
) -> None:
    results: list[PaperResult] = []

    for paper, graph in tqdm(
        zip(papers, graphs), desc="Classifying papers", total=len(papers)
    ):
        user_prompt = user_prompt_template.format(
            title=paper.title, abstract=paper.abstract, graph=graph.model_dump_json()
        )
        classified = run_gpt_classify(
            client, _CLASSIFY_SYSTEM_PROMPT, user_prompt, model
        )
        results.append(
            PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                introduction=paper.introduction,
                ratings=paper.ratings,
                sections=paper.sections,
                y_true=paper.is_approved(),
                y_pred=classified.result.approved,
            )
        )

    metrics = _calculate_metrics(results)
    logger.info(f"Metrics:\n{metrics.model_dump_json(indent=2)}")

    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    (classification_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    (classification_dir / "result.json").write_bytes(
        TypeAdapter(list[PaperResult]).dump_json(results, indent=2)
    )


def _display_graphs(
    model: str,
    graph_user_prompt_key: str,
    papers: Iterable[Paper],
    graphs: Iterable[Graph],
    output_dir: Path,
    visualise: bool,
) -> None:
    for paper, graph in zip(papers, graphs):
        dag = graph_to_dag(graph)
        dag.save(output_dir / f"{paper.title}.graphml")

        try:
            dag.visualise_hierarchy(
                show=visualise,
                img_path=output_dir / f"{paper.title}.png",
                description=f"index - model: {model} - prompt: {graph_user_prompt_key}",
            )
        except hierarchical_graph.GraphError:
            logger.exception("Error visualising graph")


def _generate_graphs(
    client: OpenAI, papers: Sequence[Paper], model: str, user_prompt: str
) -> list[Graph]:
    time_start = time.perf_counter()

    total_cost = 0
    graphs: list[Graph] = []

    for example in tqdm(papers, desc="Extracting graphs"):
        prompt = user_prompt.format(
            title=example.title,
            abstract=example.abstract,
            introduction=example.introduction,
        )
        result = run_gpt_graph(client, _GRAPH_SYSTEM_PROMPT, prompt, model)
        graph = Graph.from_gpt_graph(result.result)
        total_cost += result.cost

        sentences = (sum(e.type == EntityType.SENTENCE for e in graph.entities),)
        hierarchy_valid = graph_to_dag(graph).validate_hierarchy()
        properties_valid = validate_rules(graph)

        logger.debug(
            "Example:\n"
            f"{example}\n\n"
            "Graph:\n"
            f"{graph}\n\n"
            f"Number of sentences: {sentences}\n"
            f"Graph validation - DAG/Hierarchy: {hierarchy_valid or 'Valid'}\n"
            f"Graph validation - Properties: {properties_valid or 'Valid'}\n"
        )

        graphs.append(graph)

    logger.info(f"Total cost: ${total_cost:.10f}")

    time_elapsed = time.perf_counter() - time_start
    logger.info(f"Time elapsed: {_convert_time_elapsed(time_elapsed)}")

    return graphs


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
    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = _MODEL_SYNONYMS.get(model, model)

    _log_config(
        model=model,
        data_path=data_path,
        limit=limit,
        user_prompt=graph_user_prompt_key,
        output_dir=output_dir,
    )

    client = OpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_text())
    papers = data[:limit]

    graphs = _generate_graphs(
        client, papers, model, _GRAPH_USER_PROMPTS[graph_user_prompt_key]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    _display_graphs(model, graph_user_prompt_key, papers, graphs, output_dir, visualise)

    if classify:
        _classify_papers(
            client,
            model,
            _CLASSIFY_USER_PROMPTS[classify_user_prompt_key],
            papers,
            graphs,
            output_dir,
        )


def graph_to_dag(graph: Graph) -> hierarchical_graph.DiGraph:
    return hierarchical_graph.DiGraph.from_elements(
        nodes=[hierarchical_graph.Node(e.name, e.type.value) for e in graph.entities],
        edges=[
            hierarchical_graph.Edge(r.source, r.target, r.type.value)
            for r in graph.relationships
        ],
    )


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

    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        dest="subcommand",
        help="Additional help",
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

    args = parser.parse_args()
    setup_logging(logging.getLogger("extract_graph"))

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
        ("CLASSIFICATION PROMPTS", _CLASSIFY_USER_PROMPTS),
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
