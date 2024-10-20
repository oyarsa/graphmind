"""Generate graphs from papers from the ASAP-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Self

import dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper_hypergraph import evaluation_metrics, hierarchical_graph
from paper_hypergraph.gpt.run_gpt import GPTResult, Prompt, PromptResult, run_gpt
from paper_hypergraph.util import BlockTimer, setup_logging

logger = logging.getLogger("extract_graph")


class RelationType(StrEnum):
    SUPPORT = "support"
    CONTRAST = "contrast"


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
    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


class GPTRelationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_index: int
    target_index: int
    type: RelationType


class EntityType(StrEnum):
    TITLE = "title"
    CONCEPT = "concept"
    SENTENCE = "sentence"


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

        return "\n".join(
            [
                f"Nodes: {len(self.entities)}",
                f"Edges: {len(self.relationships)}",
                f"Titles: {sum(e.type is EntityType.TITLE for e in self.entities)}",
                f"Concepts: {sum(e.type is EntityType.CONCEPT for e in self.entities)}",
                f"Sentences: {sum(e.type is EntityType.SENTENCE for e in self.entities)}",
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
    def from_gpt_graph(cls, gpt_graph: GPTGraph) -> Self:
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
                f"Titles: {sum(e.type is EntityType.TITLE for e in self.entities)}",
                f"Concepts: {sum(e.type is EntityType.CONCEPT for e in self.entities)}",
                f"Sentences: {sum(e.type is EntityType.SENTENCE for e in self.entities)}",
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
# Include the synonyms and their keys in the allowed models
_MODELS_ALLOWED = sorted(_MODEL_SYNONYMS.keys() | _MODEL_SYNONYMS.values())


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


class GPTClassify(BaseModel):
    model_config = ConfigDict(frozen=True)

    rationale: str
    approved: bool


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

_GRAPH_USER_PROMPTS = {
    # FIX: Update variables to match the actual data
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
- Relations can be of two types: supporting or contrasting.
- There MUST be at least two sentences of each type.
- Supporting relations support the key concepts, provide evidence of why they might be \
true or explain them. For example, they can be supporting sentences from citations, \
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

Main text:
{main_text}


#####
Output:
""",
}


RATING_APPROVAL_THRESHOLD = 5
"""A rating is an approval if it's greater of equal than this."""


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
            Graph.from_gpt_graph(result.result)
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


class Paper(BaseModel):
    """ASAP-Review paper with only currently useful fields.

    Check the ASAP-Review dataset to see what else is available, and use
    paper_hypergraph.asap.extract and asap.filter to add them to this dataset.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]

    def is_approved(
        self, strategy: RatingEvaluationStrategy = RatingEvaluationStrategy.MEAN
    ) -> bool:
        return strategy.is_approved(self.ratings)

    def main_text(self) -> str:
        return "\n".join(s.text for s in self.sections)

    def __str__(self) -> str:
        main_text_words_num = len(self.main_text().split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {self.ratings}\n"
        )


class PaperResult(Paper):
    """ASAP-Review dataset paper with added approval ground truth and GPT prediction."""

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
) -> GPTResult[list[PaperResult]]:
    """Classify Papers into approved/not approved using the generated graphs.

    The output - input dataset information, predicted values and metrics - is written
    to {output_dir}/classification.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt_template: User prompt template to use for classification to be filled
        papers: Papers from the ASAP-Review dataset to classify
        graphs: Graphs generated from the papers
        output_dir: Directory to save the classification results

    Returns:
        None. The outputs are saved to disk.
    """
    results: list[PaperResult] = []
    total_cost = 0

    for paper, graph in tqdm(
        zip(papers, graphs), desc="Classifying papers", total=len(papers)
    ):
        user_prompt = user_prompt_template.format(
            title=paper.title,
            abstract=paper.abstract,
            graph=graph.model_dump_json(),
        )
        result = run_gpt(
            GPTClassify, client, _CLASSIFY_SYSTEM_PROMPT, user_prompt, model
        )
        total_cost += result.cost
        classified = result.result

        results.append(
            PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                ratings=paper.ratings,
                sections=paper.sections,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
            )
        )

    return GPTResult(results, total_cost)


def _display_and_save_graphs(
    model: str,
    graph_user_prompt_key: str,
    papers: Iterable[Paper],
    graph_results: Iterable[PromptResult[Graph]],
    output_dir: Path,
    visualise: bool,
) -> None:
    """Save generated graphs and plot them to PNG files and (optionally) the screen.

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
        dag = graph_to_dag(graph_result.item)
        # TODO: Merge these two files into one. Maybe have the GraphML be a field in
        # an output JSON.
        dag.save(output_dir / f"{paper.title}.graphml")
        (output_dir / f"{paper.title}_prompt.json").write_text(
            graph_result.prompt.model_dump_json()
        )

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

    if model not in _MODELS_ALLOWED:
        raise ValueError(f"Model {model} not in allowed models: {_MODELS_ALLOWED}")
    model = _MODEL_SYNONYMS.get(model, model)

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

    with BlockTimer() as timer_gen:
        graph_results = _generate_graphs(client, papers, model, graph_user_prompt)

    logger.info(f"Graph generation time elapsed: {timer_gen.human}")
    logger.info(f"Total graph generation cost: ${graph_results.cost:.10f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _display_and_save_graphs(
        model,
        graph_user_prompt_key,
        papers,
        graph_results.result,
        output_dir,
        visualise,
    )

    if classify:
        classify_user_prompt = _CLASSIFY_USER_PROMPTS[classify_user_prompt_key]

        with BlockTimer() as timer_class:
            graphs = [result.item for result in graph_results.result]
            results = _classify_papers(
                client, model, classify_user_prompt, papers, graphs
            )
        metrics = _calculate_metrics(results.result)
        logger.info(f"Metrics:\n{metrics.model_dump_json(indent=2)}")

        logger.info(f"Classification time elapsed: {timer_class.human}")
        logger.info(f"Total classification cost: ${results.cost:.10f}")

        classification_dir = output_dir / "classification"
        classification_dir.mkdir(parents=True, exist_ok=True)

        (classification_dir / "metrics.json").write_text(
            metrics.model_dump_json(indent=2)
        )
        (classification_dir / "result.json").write_bytes(
            TypeAdapter(list[PaperResult]).dump_json(results.result, indent=2)
        )


def graph_to_dag(graph: Graph) -> hierarchical_graph.DiGraph:
    return hierarchical_graph.DiGraph.from_elements(
        nodes=[hierarchical_graph.Node(e.name, e.type.value) for e in graph.entities],
        edges=[
            hierarchical_graph.Edge(r.source, r.target, r.type.value)
            for r in graph.relationships
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
        choices=_MODELS_ALLOWED,
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
    setup_logging(logger)

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


if __name__ == "__main__":
    main()
