"""Common types used to represent entities in the GPT-based extraction tools."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self, override

from pydantic import BaseModel, ConfigDict, Field, computed_field

import paper.semantic_scholar as s2
from paper import hierarchical_graph, peerread
from paper.util import hashstr
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper import peter


class EntityType(StrEnum):
    """Entity type in hierarchical graph."""

    TITLE = "title"
    TLDR = "tldr"
    PRIMARY_AREA = "primary_area"
    KEYWORD = "keyword"
    CLAIM = "claim"
    METHOD = "method"
    EXPERIMENT = "experiment"


class Relationship(BaseModel):
    """Relationship between nodes in the hierarchical graph."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str


class Entity(BaseModel):
    """Entity in the hierarchical graph."""

    model_config = ConfigDict(frozen=True)

    label: str
    type: EntityType
    detail: str | None = None


class Graph(Record):
    """GPT output graph representing the hierarchical graph.

    See `valid_status` for the rules governing this graph.
    """

    title: str
    abstract: str
    entities: Sequence[Entity]
    relationships: Sequence[Relationship]

    @property
    def id(self) -> str:
        """Identify graph from the title and abstract of the paper that generated it."""
        return hashstr(self.title + self.abstract)

    @computed_field
    @property
    def valid_status(self) -> str:
        """Check if graph rules hold. Returns first error message if invalid, or "Valid".

        See `valid_status_all` for all messages, if there's more than one error.

        Returns:
            Error message describing the rule violated if the graph is invalid.
            "Valid" if the graph is follows all rules.
            "Empty graph" is the graph has no entities.
        """
        return self.valid_status_all[0]

    @computed_field
    @property
    def valid_status_all(self) -> list[str]:
        """Check if graph rules hold. Returns all error messages if invalid, or "Valid".

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
            Error messages describing the rules violated if the graph is invalid.
            ["Valid"] if the graph is follows all rules.
            ["Empty graph"] is the graph has no entities.
        """
        if not self.entities:
            return ["Empty graph"]

        errors: list[str] = []

        # Rule 0: Every entity must be unique
        if entity_counts := [
            f"{entity} ({count})"
            for entity, count in Counter(e.label for e in self.entities).most_common()
            if count > 1
        ]:
            errors.append(f"Entities with non-unique names: {", ".join(entity_counts)}")

        entities = {entity.label: entity for entity in self.entities}
        incoming: defaultdict[str, list[Relationship]] = defaultdict(list)
        outgoing: defaultdict[str, list[Relationship]] = defaultdict(list)

        for relation in self.relationships:
            incoming[relation.target].append(relation)
            outgoing[relation.source].append(relation)

        # Rule 1: Exactly one node from Title, Primary Area and TLDR
        singletons = [EntityType.TITLE, EntityType.PRIMARY_AREA, EntityType.TLDR]
        for node_type in singletons:
            nodes = _get_nodes_of_type(self.entities, node_type)
            if len(nodes) != 1:
                errors.append(
                    f"Found {len(nodes)} '{node_type}' nodes. Should be exactly 1."
                )

        # Rule 2: Title node cannot have incoming edges
        title = _get_nodes_of_type(self.entities, EntityType.TITLE)[0]
        if incoming[title.label]:
            errors.append("Title node should not have any incoming edges.")

        # Rule 3: TLDR, Primary Area and Keyword nodes only have incoming edges from Title
        level_2 = [EntityType.TLDR, EntityType.PRIMARY_AREA, EntityType.KEYWORD]
        for node_type in level_2:
            nodes = _get_nodes_of_type(self.entities, node_type)
            for node in nodes:
                inc_edges = incoming[node.label]
                if len(inc_edges) != 1:
                    errors.append(
                        f"Found {len(inc_edges)} incoming edges to node type '{node_type}'."
                        f" Should be exactly 1. Node: '{node.label}'"
                    )

                inc_node = entities[inc_edges[0].source]
                if inc_node.type is not EntityType.TITLE:
                    errors.append(
                        f"Incoming edge to '{node_type}' is not Title, but '{inc_node.type}'."
                    )

        # Rule 4: Primary Area, Keyword and Experiment nodes don't have outgoing edges
        level_leaf = [
            EntityType.PRIMARY_AREA,
            EntityType.KEYWORD,
            EntityType.EXPERIMENT,
        ]
        for node_type in level_leaf:
            for node in _get_nodes_of_type(self.entities, node_type):
                if out := outgoing[node.label]:
                    errors.append(
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
            for node in _get_nodes_of_type(self.entities, cur_type):
                for edge in outgoing[node.label]:
                    type_ = entities[edge.target].type
                    if type_ is not next_type:
                        errors.append(
                            f"Found illegal outgoing edge from '{cur_type}' to '{type_}'"
                        )

        # Incoming edges
        for prev_type, cur_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self.entities, cur_type):
                for edge in incoming[node.label]:
                    type_ = entities[edge.source].type
                    if type_ is not prev_type:
                        errors.append(
                            f"Found illegal incoming edge from '{type_}' to '{cur_type}'"
                        )

        # Rule 6: At mid levels, each node in a level must be connected to at least one
        # node in the previous level.
        for prev_type, cur_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self.entities, cur_type):
                inc = [
                    edge
                    for edge in incoming[node.label]
                    if entities[edge.source].type is prev_type
                ]
                if not inc:
                    errors.append(
                        f"Node type '{cur_type}' has no incoming edges from '{prev_type}'."
                        " Should be at least 1."
                    )
        # Rule 7: At mid levels, each node in a level must be connected to at least one
        # node in the next level.
        for cur_type, next_type in itertools.pairwise(level_order):
            for node in _get_nodes_of_type(self.entities, cur_type):
                out = [
                    edge
                    for edge in outgoing[node.label]
                    if entities[edge.target].type is next_type
                ]
                if not out:
                    errors.append(
                        f"Node type '{cur_type}' has no outgoing edges to '{next_type}'."
                        " Should be at least 1."
                    )

        # Rule 8: No cycles
        if self.to_digraph().has_cycle():
            errors.append("Graph has cycles")

        if errors:
            return errors
        return ["Valid"]

    def __str__(self) -> str:
        """Display the entities, relationships, counts and validity of the graph."""
        type_index = list(EntityType).index
        entities = "\n".join(
            f"  {i}. {c.type} - {c.label}"
            for i, c in enumerate(
                sorted(self.entities, key=lambda e: (type_index(e.type), e.label)), 1
            )
        )

        entity_type = {e.label: e.type for e in self.entities}
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

    @classmethod
    def empty(cls) -> Self:
        """Graph without entities or relatioships.

        Used in cases where it's not possible to extract a valid graph.
        """
        return cls(
            title="",
            abstract="",
            entities=[],
            relationships=[],
        )

    def to_text(self) -> str:
        """Convert graph to LLM-readable text.

        Sorts the entities topologically, then creates paragraphs with each entity's
        type, name and description, if available.

        If the graph is empty, returns an empty string.
        """
        return self.to_digraph().to_text()

    def to_digraph(self) -> hierarchical_graph.DiGraph:
        """Convert to a proper hierarchical graph."""
        return hierarchical_graph.DiGraph.from_elements(
            nodes=[
                hierarchical_graph.Node(e.label, e.type.value, e.detail)
                for e in self.entities
            ],
            edges=[
                hierarchical_graph.Edge(r.source, r.target) for r in self.relationships
            ],
        )


def _get_nodes_of_type(entities: Iterable[Entity], type_: EntityType) -> list[Entity]:
    return [e for e in entities if e.type is type_]


class PaperSection(BaseModel):
    """Section of a full paper with its headin and content text."""

    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


class Paper(Record):
    """PeerRead paper with only currently useful fields."""

    title: str
    abstract: str
    reviews: Sequence[peerread.PaperReview]
    authors: Sequence[str]
    sections: Sequence[PaperSection]
    rationale: str
    rating: int


class ReviewEvaluation(BaseModel):
    """Peer review with its original rating and predicted rating from GPT."""

    model_config = ConfigDict(frozen=True)

    # Original review data
    rating: Annotated[
        int,
        Field(description="Novelty rating given by the reviewer (1 to 5)"),
    ]
    confidence: Annotated[int | None, Field(description="Confidence from the reviewer")]
    rationale: Annotated[str, Field(description="Explanation given for the rating")]

    # Predicted data
    extracted_rationale: Annotated[
        str | None, Field(description="Novelty rationale extracted.")
    ] = None
    predicted_rating: Annotated[
        int | None,
        Field(description="Predicted novelty rating from GPT (1 to 5)"),
    ] = None
    predicted_rationale: Annotated[
        str | None,
        Field(description="GPT's explanation for the predicted rating"),
    ] = None


class PaperWithReviewEval(Record):
    """PeerRead paper with predicted reviews."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[ReviewEvaluation], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[peerread.PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    references: Annotated[
        Sequence[peerread.PaperReference],
        Field(description="References made in the paper"),
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]
    review: Annotated[ReviewEvaluation, Field(description="Main review for the paper")]
    rationale: Annotated[str, Field(description="Rationale for the main review")]
    rating: Annotated[int, Field(description="Rating (1-5) for the main review")]

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    def main_text(self) -> str:
        """Join all paper sections to form the main text."""
        return "\n".join(s.text for s in self.sections)

    def __str__(self) -> str:
        """Display title, abstract, rating scores and count of words in main text."""
        main_text_words_num = len(self.main_text().split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {[r.rating for r in self.reviews]}\n"
        )


class Prompt(BaseModel):
    """Prompt used in a GPT API request."""

    model_config = ConfigDict(frozen=True)

    system: str
    user: str


class PromptResult[T](BaseModel):
    """Wrapper around a GPT API response with the full prompt that generated it."""

    model_config = ConfigDict(frozen=True)

    item: T
    prompt: Prompt

    @classmethod
    def unwrap[U](cls, data: Iterable[PromptResult[U]]) -> list[U]:
        """Transform an iterable of wrapped items into a list of the internal elements."""
        return [x.item for x in data]

    def map[U](self, fn: Callable[[T], U]) -> PromptResult[U]:
        """Apply function to wrapped `item`."""
        return PromptResult(item=fn(self.item), prompt=self.prompt)


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
        """Return instance with all empty items."""
        return cls(tasks=(), methods=(), metrics=(), resources=(), relations=())

    def is_valid(self) -> bool:
        """Check if relations and at least two term lists are non-empty."""
        if not self.relations:
            return False

        term_lists = [self.tasks, self.methods, self.metrics, self.resources]
        return sum(bool(term_list) for term_list in term_lists) >= 2


type PaperToAnnotate = s2.Paper | s2.PaperWithS2Refs


class PaperAnnotated(Record):
    """`PaperToAnnotate` with its annotated key terms. Includes GPT prompts used."""

    terms: PaperTerms
    paper: PaperToAnnotate
    background: str
    target: str

    @property
    def id(self) -> str:
        """Identify annotated paper by the underlying paper ID."""
        return self.paper.id

    def is_valid(self) -> bool:
        """Check that `terms` are valid, and `background` and `target` are non-empty.

        For `terms`, see GPTTerms.is_valid.
        """
        return self.terms.is_valid() and bool(self.background) and bool(self.target)

    def target_terms(self) -> list[str]:
        """Get unique target terms (tasks) from the paper."""
        return sorted(set(self.terms.tasks))

    @property
    def title(self) -> str:
        """Title of the underlying paper, or '<unknown>' if absent."""
        return self.paper.title or "<unknown>"

    @property
    def abstract(self) -> str:
        """Abstract of the underlying paper, or '<unknown>' if absent."""
        return self.paper.abstract or "<unknown>"


class PeerReadAnnotated(Record):
    """PeerRead Paper with its annotated key terms. Includes GPT prompts used."""

    terms: PaperTerms
    paper: s2.PaperWithS2Refs
    background: str
    target: str

    @property
    def id(self) -> str:
        """Identify annotated PeerRead paper by the underlying ID."""
        return self.paper.id

    def target_terms(self) -> list[str]:
        """Get unique target terms (tasks) from the paper."""
        return sorted(set(self.terms.tasks))

    @property
    def title(self) -> str:
        """Title of the underlying paper."""
        return self.paper.title

    @property
    def abstract(self) -> str:
        """Abstract of the underlying paper."""
        return self.paper.abstract


class PaperWithRelatedSummary(Record):
    """PeerRead paper with its related papers formatted as prompt input."""

    paper: PeerReadAnnotated
    related: Sequence[PaperRelatedSummarised]

    @property
    def id(self) -> str:
        """Identify graph result as the underlying paper's ID."""
        return self.paper.id

    @property
    def title(self) -> str:
        """Title of the underlying paper."""
        return self.paper.title

    @property
    def abstract(self) -> str:
        """Abstract of the underlying paper."""
        return self.paper.abstract


class PaperRelatedSummarised(Record):
    """PETER-related paper with summary."""

    summary: str

    paper_id: str
    title: str
    abstract: str
    score: float
    polarity: peerread.ContextPolarity

    @property
    def id(self) -> str:
        """Identify the summary by its underlying paper ID."""
        return self.paper_id

    @classmethod
    def from_related(cls, related: peter.PaperRelated, summary: str) -> Self:
        """PETER-related paper with generated summary."""
        return cls(
            summary=summary,
            paper_id=related.paper_id,
            title=related.title,
            abstract=related.abstract,
            score=related.score,
            polarity=peerread.ContextPolarity(related.polarity),
        )
