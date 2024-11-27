"""Common types used to represent entities in the GPT-based extraction tools."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

import paper.external_data.semantic_scholar as s2
from paper import asap, hierarchical_graph
from paper.util import hashstr
from paper.util.serde import Record


class EntityType(StrEnum):
    TITLE = "title"
    TLDR = "tldr"
    PRIMARY_AREA = "primary_area"
    KEYWORD = "keyword"
    CLAIM = "claim"
    METHOD = "method"
    EXPERIMENT = "experiment"


class Relationship(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: EntityType


class Graph(Record):
    title: str
    abstract: str
    entities: Sequence[Entity]
    relationships: Sequence[Relationship]

    @property
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    @computed_field
    @property
    def valid_status(self) -> str:
        """Check if graph rules hold. Returns error message if invalid, or "Valid".

        Rules:
        0. All entity names must be unique.
        1. There must be exactly one node of types Title, Primary Area and TLDR.
        2. The Title node cannot have incoming edges.
        3. In the second level, TLDR, Primary Area and Keyword nodes can only have one
           incoming each, and it must be from Title.
        4. Primary Area, Keyword and Experiment nodes don't have outgoing edges.
        5. At later mid levels, nodes can only have incoming edges from the previous
           level and outgoing edges to the next level.
           The sequence in levels is:
            1. Title
            2. Title -> Primary Area, Keywords, TLDR
            3. TLDR -> Claims
            4. Claims -> Methods
            5. Methods -> Experiments
        6. At mid levels, in each node in a level must be connected to at least one node
           from the previous level.
        7. At mid levels, in each node in a level must be connected to at least one node
           from the next level.
        8. There should be no cycles

        Note: this function doesn't throw an exception if the graph is invalid, it just
        returns the error message. The graph is allowed to be invalid, but it's useful to
        know why it's invalid.

        Returns:
            Error message describing the rule violated if the graph is invalid.
            "Valid" if the graph is follows all rules.
        """
        # Rule 0: Every entity must be unique
        if entity_counts := [
            f"{entity} ({count})"
            for entity, count in Counter(e.name for e in self.entities).most_common()
            if count > 1
        ]:
            return f"Entities with non-unique names: {", ".join(entity_counts)}"

        entities = {entity.name: entity for entity in self.entities}
        incoming: defaultdict[str, list[Relationship]] = defaultdict(list)
        outgoing: defaultdict[str, list[Relationship]] = defaultdict(list)

        for relation in self.relationships:
            incoming[relation.target].append(relation)
            outgoing[relation.source].append(relation)

        # Rule 1: Exactly one node from Title, Primary Area and TLDR
        singletons = [EntityType.TITLE, EntityType.PRIMARY_AREA, EntityType.TLDR]
        for node_type in singletons:
            nodes = _get_nodes_of_type(self, node_type)
            if len(nodes) != 1:
                return f"Found {len(nodes)} '{node_type}' nodes. Should be exactly 1."

        # Rule 2: Title node cannot have incoming edges
        title = _get_nodes_of_type(self, EntityType.TITLE)[0]
        if incoming[title.name]:
            return "Title node should not have any incoming edges."

        # Rule 3: TLDR, Primary Area and Keyword nodes only have incoming edges from Title
        level_2 = [EntityType.TLDR, EntityType.PRIMARY_AREA, EntityType.KEYWORD]
        for node_type in level_2:
            nodes = _get_nodes_of_type(self, node_type)
            for node in nodes:
                inc_edges = incoming[node.name]
                if len(inc_edges) != 1:
                    return (
                        f"Found {len(inc_edges)} incoming edges to node type '{node_type}'."
                        f" Should be exactly 1. Node: '{node.name}'"
                    )

                inc_node = entities[inc_edges[0].source]
                if inc_node.type is not EntityType.TITLE:
                    return f"Incoming edge to '{node_type}' is not Title, but '{inc_node.type}'."

        # Rule 4: Primary Area, Keyword and Experiment nodes don't have outgoing edges
        level_leaf = [
            EntityType.PRIMARY_AREA,
            EntityType.KEYWORD,
            EntityType.EXPERIMENT,
        ]
        for node_type in level_leaf:
            for node in _get_nodes_of_type(self, node_type):
                if out := outgoing[node.name]:
                    return (
                        f"Found {len(out)} outgoing edges from node type '{node_type}'."
                        " Should be 0."
                    )

        # Rule 5: At mid levels, edges come from the previous level and go to the next
        level_order = [
            EntityType.TLDR,
            EntityType.CLAIM,
            EntityType.METHOD,
            EntityType.EXPERIMENT,
        ]
        # Outgoing edges
        for cur_type, next_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self, cur_type):
                for edge in outgoing[node.name]:
                    type_ = entities[edge.target].type
                    if type_ is not next_type:
                        return f"Found illegal outgoing edge from '{cur_type}' to '{type_}'"

        # Incoming edges
        for prev_type, cur_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self, cur_type):
                for edge in incoming[node.name]:
                    type_ = entities[edge.source].type
                    if type_ is not prev_type:
                        return f"Found illegal incoming edge from '{type_}' to '{cur_type}'"

        # Rule 6: At mid levels, each node in a level must be connected to at least one
        # node in the previous level.
        for prev_type, cur_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self, cur_type):
                inc = [
                    edge
                    for edge in incoming[node.name]
                    if entities[edge.source].type is prev_type
                ]
                if not inc:
                    return (
                        f"Node type '{cur_type}' has no incoming edges from '{prev_type}'."
                        " Should be at least 1."
                    )
        # Rule 7: At mid levels, each node in a level must be connected to at least one
        # node in the next level.
        for cur_type, next_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self, cur_type):
                out = [
                    edge
                    for edge in outgoing[node.name]
                    if entities[edge.target].type is next_type
                ]
                if not out:
                    return (
                        f"Node type '{cur_type}' has no outgoing edges to '{next_type}'."
                        " Should be at least 1."
                    )

        # Rule 8: No cycles
        if graph_to_digraph(self).has_cycle():
            return "Graph has cycles"

        return "Valid"

    def __str__(self) -> str:
        type_index = list(EntityType).index
        entities = "\n".join(
            f"  {i}. {c.type} - {c.name}"
            for i, c in enumerate(
                sorted(self.entities, key=lambda e: (type_index(e.type), e.name)), 1
            )
        )

        entity_type = {e.name: e.type for e in self.entities}
        relationships = "\n".join(
            f"{i}. {entity_type[r.source]} -> {entity_type[r.target]}\n"
            f"- {r.source}\n"
            f"- {r.target}\n"
            for i, r in enumerate(
                sorted(
                    self.relationships,
                    key=lambda r: (
                        type_index(entity_type[r.source]),
                        type_index(entity_type[r.target]),
                        r.source,
                        r.target,
                    ),
                ),
                1,
            )
        )
        node_type_counter = Counter(e.type for e in self.entities)
        node_type_counts = sorted(
            ((k, node_type_counter.get(k, 0)) for k in EntityType),
            key=lambda x: type_index(x[0]),
        )

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
                f"Validation: {self.valid_status}",
                "",
            ]
        )


def graph_to_digraph(graph: Graph) -> hierarchical_graph.DiGraph:
    return hierarchical_graph.DiGraph.from_elements(
        nodes=[hierarchical_graph.Node(e.name, e.type.value) for e in graph.entities],
        edges=[
            hierarchical_graph.Edge(r.source, r.target) for r in graph.relationships
        ],
    )


def _get_nodes_of_type(graph: Graph, type_: EntityType) -> list[Entity]:
    return [e for e in graph.entities if e.type is type_]


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


RATING_APPROVAL_THRESHOLD = 5
"""A rating is an approval if it's greater of equal than this."""


class RatingEvaluationStrategy(StrEnum):
    MEAN = "mean"
    """Mean rating is higher than the threshold."""
    MAJORITY = "majority"
    """Majority of ratings are higher than the threshold."""
    DECISION = "decision"
    """Use the provided approval decision."""
    DEFAULT = DECISION

    def is_approved(self, decision: bool, ratings: Sequence[int]) -> bool:
        match self:
            case RatingEvaluationStrategy.DECISION:
                return decision
            case RatingEvaluationStrategy.MEAN:
                mean = sum(ratings) / len(ratings)
                return mean >= RATING_APPROVAL_THRESHOLD
            case RatingEvaluationStrategy.MAJORITY:
                approvals = [r >= RATING_APPROVAL_THRESHOLD for r in ratings]
                return sum(approvals) >= len(approvals) / 2


class Paper(Record):
    """ASAP-Review paper with only currently useful fields.

    Check the ASAP-Review dataset to see what else is available, and use
    paper.asap.extract and asap.filter to add them to this dataset.
    """

    title: str
    abstract: str
    reviews: Sequence[asap.PaperReview]
    sections: Sequence[PaperSection]
    approval: bool

    @property
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    def is_approved(
        self, strategy: RatingEvaluationStrategy = RatingEvaluationStrategy.DEFAULT
    ) -> bool:
        return strategy.is_approved(self.approval, [r.rating for r in self.reviews])

    def main_text(self) -> str:
        return "\n".join(s.text for s in self.sections)

    def __str__(self) -> str:
        main_text_words_num = len(self.main_text().split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {[r.rating for r in self.reviews]}\n"
        )


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    user: str


class PromptResult[T](BaseModel):
    model_config = ConfigDict(frozen=True)

    item: T
    prompt: Prompt

    @classmethod
    def unwrap[U](cls, data: Iterable[PromptResult[U]]) -> list[U]:
        return [x.item for x in data]


class PaperGraph(Record):
    paper: Paper
    graph: PromptResult[Graph]

    @computed_field
    @property
    def id(self) -> str:
        return self.paper.id

    @model_validator(mode="after")
    def validate_matching_ids(self) -> Self:
        if self.paper.id != self.graph.item.id:
            raise ValueError("Paper ID must match graph item ID")
        return self


class S2Paper(Record):
    """Paper returned by the Semantic Scholar API. Everything's optional but `paperId`.

    This is to avoid validation errors in the middle of the download. We'll only save
    those with non-empty `abstract`, though.
    """

    # Semantic Scholar's primary unique identifier for a paper.
    paper_id: Annotated[str, Field(alias="paperId")]
    # Semantic Scholar's secondary unique identifier for a paper.
    corpus_id: Annotated[int | None, Field(alias="corpusId")]
    # URL of the paper on the Semantic Scholar website.
    url: str
    # Title of the paper.
    title: str
    # The paper's abstract. Note that due to legal reasons, this may be missing even if
    # we display an abstract on the website.
    abstract: str
    # The year the paper was published.
    year: int
    # The total number of papers this paper references.
    reference_count: Annotated[int, Field(alias="referenceCount")]
    # The total number of papers that reference this paper.
    citation_count: Annotated[int, Field(alias="citationCount")]
    # A subset of the citation count, where the cited publication has a significant
    # impact on the citing publication.
    influential_citation_count: Annotated[int, Field(alias="influentialCitationCount")]
    # The tldr paper summary.
    tldr: s2.Tldr | None = None
    # Paper authors.
    authors: Sequence[s2.Author]

    @property
    def id(self) -> str:
        return self.paper_id


class PaperTermRelation(BaseModel):
    """Represents a directed 'used for' relation between two scientific terms.

    Relations are always (head, used-for, tail).
    """

    model_config = ConfigDict(frozen=True)

    head: Annotated[str, Field(description="Head term of the relation.")]
    tail: Annotated[str, Field(description="Tail term of the relation.")]


class PaperTerms(BaseModel):
    """Structured output for scientific term extraction."""

    model_config = ConfigDict(frozen=True)

    tasks: Annotated[
        Sequence[str],
        Field(description="Core problems, objectives or applications addressed."),
    ]
    methods: Annotated[
        Sequence[str],
        Field(
            description="Technical approaches, algorithms, or frameworks used/proposed."
        ),
    ]
    metrics: Annotated[
        Sequence[str], Field(description="Evaluation metrics and measures mentioned.")
    ]
    resources: Annotated[
        Sequence[str], Field(description="Datasets, resources, or tools utilised.")
    ]
    relations: Annotated[
        Sequence[PaperTermRelation],
        Field(description="Directed relations between terms."),
    ]

    @classmethod
    def empty(cls) -> Self:
        return cls(tasks=(), methods=(), metrics=(), resources=(), relations=())

    def is_valid(self) -> bool:
        """Check if relations and at least two term lists are non-empty."""
        if not self.relations:
            return False

        term_lists = [self.tasks, self.methods, self.metrics, self.resources]
        return sum(bool(term_list) for term_list in term_lists) >= 2


type PaperToAnnotate = S2Paper | Paper


class PaperAnnotated(Record):
    """`PaperToAnnotate` with its annotated key terms. Includes GPT prompts used."""

    terms: PaperTerms
    paper: PaperToAnnotate
    background: str
    target: str

    @property
    def id(self) -> str:
        return self.paper.id

    def is_valid(self) -> bool:
        """Check that `terms` are valid, and `background` and `target` are non-empty.

        For `terms`, see GPTTerms.is_valid.
        """
        return self.terms.is_valid() and bool(self.background) and bool(self.target)

    def target_terms(self) -> list[str]:
        """Get target terms from the paper, i.e. unique tail nodes from the relations."""
        return sorted({r.tail for r in self.terms.relations})


class ASAPAnnotated(Record):
    """ASAP Paper with its annotated key terms. Includes GPT prompts used."""

    terms: PaperTerms
    paper: Paper
    background: str
    target: str

    @property
    def id(self) -> str:
        return self.paper.id
