"""Common types used to represent entities in the GPT-based extraction tools."""

from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self, override

from pydantic import Field, computed_field

import paper.semantic_scholar as s2
from paper import hierarchical_graph
from paper import peerread as pr
from paper.types import Immutable, PaperProxy
from paper.util import (
    fix_spaces_before_punctuation,
    format_numbered_list,
    hashstr,
    on_exception,
    remove_parenthetical,
)
from paper.util.serde import Record

if TYPE_CHECKING:
    from paper import related_papers as rp

logger = logging.getLogger(__name__)

RATIONALE_ERROR = "<error>"


def is_rationale_valid(rationale: str) -> bool:
    """Check if rationale is valid. Errors in extraction generate the `RATIONALE_ERROR`."""
    return rationale != RATIONALE_ERROR


class EntityType(StrEnum):
    """Entity type in hierarchical graph."""

    TITLE = "title"
    TLDR = "tldr"
    PRIMARY_AREA = "primary_area"
    KEYWORD = "keyword"
    CLAIM = "claim"
    METHOD = "method"
    EXPERIMENT = "experiment"


class Relationship(Immutable):
    """Relationship between nodes in the hierarchical graph."""

    source: str
    target: str


class Excerpt(Immutable):
    """Text from the paper where the entity is mentioned."""

    section: Annotated[
        str,
        Field(description="Section (and nested subsections) where the text appears."),
    ]
    text: Annotated[
        str, Field(description="Text that mentions the entity, copied verbatim.")
    ]


class Entity(Immutable):
    """Entity in the hierarchical graph."""

    label: str
    type: EntityType
    detail: str | None = None
    excerpts: Sequence[Excerpt] | None = None

    def __hash__(self) -> int:
        """Entity has is the hash of its members."""
        return hash((self.label, self.type, self.detail))


class LinearisationMethod(StrEnum):
    """How to convert a `Graph` into LLM-readable text."""

    TOPO = "topo"
    """Use topological sort to order entities and convert to text.

    Sorts the entities topologically, then creates paragraphs with each entity's
    type, name and description, if available.
    """
    FLUENT = "fluent"
    """Use a fluent template-based method to convert entities to natural text.

    The goal is for the output to look more like a crafted summary.
    """


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
            errors.append(f"Entities with non-unique names: {', '.join(entity_counts)}")

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

        return "\n".join([
            f"Nodes: {len(self.entities)}",
            f"Edges: {len(self.relationships)}",
            f"Node types: {', '.join(f'{k}: {v}' for k, v in node_type_counts)}",
            "",
            "Entities:",
            entities,
            "",
            "Relationships:",
            relationships,
            "",
            f"Validation: {self.valid_status}",
            "",
        ])

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

    def is_empty(self) -> bool:
        """Return True if the graph is empty. See also `Graph.empty`."""
        return not self.title

    @on_exception(default=RATIONALE_ERROR, logger=logger)
    def to_text(self, method: LinearisationMethod = LinearisationMethod.TOPO) -> str:
        """Convert graph to LLM-readable text using the linearisation `method`.

        If the graph is empty, returns an empty string.

        Returns:
            The graph converted to text. If the graph is invalid and cannot be converted,
            returns "<error>".

        Raises:
            Never. Any `Exception`-based errors are caught, and `<error>` is returned
            instead.
        """
        match method:
            case LinearisationMethod.TOPO:
                return self.to_digraph().to_text()
            case LinearisationMethod.FLUENT:
                return self.fluent_linearization()

    def fluent_linearization(self) -> str:
        """Convert graph to text using "fluent", natural flow.

        The conversion is template-based, so it will still be a little weird.

        Returns:
            The graph converted to text. If the graph is invalid and cannot be converted,
            returns "<error>".

        Raises:
            IndexError, KeyError or ValueError if an expected property the graph isn't
            fulfilled (e.g. no title or primary area).
        """
        entity_map = {e.label: e for e in self.entities}

        adjacent: dict[str, list[str]] = defaultdict(list)
        for rel in self.relationships:
            adjacent[rel.source].append(rel.target)

        title = _get_nodes_of_type(self.entities, EntityType.TITLE)[0]
        primary_area = _get_nodes_of_type(self.entities, EntityType.PRIMARY_AREA)[0]

        primary_text = remove_parenthetical(primary_area.label)

        sections = [
            f"This paper is titled '{title.label}'. It's about {primary_text}."
            " The key contributions are:"
        ]

        claim_sections: list[str] = []
        for claim_idx, claim in enumerate(
            _get_nodes_of_type(self.entities, EntityType.CLAIM), start=1
        ):
            methods = adjacent.get(claim.label)
            if not methods:
                continue

            method_sentences: list[str] = []
            for method_idx, method_label in enumerate(methods, start=1):
                method = entity_map.get(method_label)
                if method is None or method.type is not EntityType.METHOD:
                    continue

                experiment_labels = adjacent.get(method.label)
                if not experiment_labels:
                    continue

                experiments = [
                    _format_entity_detail_sentence(exp)
                    for label in experiment_labels
                    if (exp := entity_map.get(label))
                    and exp.type is EntityType.EXPERIMENT
                ]
                experiments_bullets = format_numbered_list(
                    experiments, prefix=f"{claim_idx}.{method_idx}.", indent=4
                )
                method_sentences.append(
                    f"{_format_entity_detail_sentence(method)}"
                    " This method is validated by these experiments:\n"
                    f"{experiments_bullets}"
                )

            claim_sections.append(
                "\n".join([
                    f"{_format_entity_detail_sentence(claim)} This is done with:",
                    format_numbered_list(
                        method_sentences, prefix=f"{claim_idx}.", indent=2
                    ),
                ])
            )

        if claim_sections:
            sections.append(format_numbered_list(claim_sections, sep="\n\n"))

        return fix_spaces_before_punctuation("\n\n".join(sections))

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


def capitalise(text: str) -> str:
    """Capitalise first letter of the string. The rest is left as-is.

    This is different from `str.capitalize` because that lowercases the rest of the
    string, but we don't.
    """
    return text[0].upper() + text[1:]


def _ensure_punctuation(text: str) -> str:
    """Add period at the end of `text` if it doesn't end with `.!?;`."""
    if text[-1] not in {".", "!", "?", ";"}:
        return f"{text}."
    return text


def _strip_punctuation(text: str) -> str:
    """Remove punctuation at the end of `text`."""
    return text.rstrip(".:,?!")


def _format_entity_detail_sentence(entity: Entity) -> str:
    """Format entity with label and detail.

    Args:
        entity: An entity of a type that has detail text (method, claim, experiment).

    Returns:
        Formatted sentence with `label: detail`, where `label` is capitalised and
        `detail` ends with a period.

    Raises:
        ValueError: if the entity doesn't contain valid detail text (None or empty).
    """
    if entity.detail is None or not entity.detail.strip():
        raise ValueError(
            f"Entity of type '{entity.type}' does not have valid detail text."
        )

    text = f"{_strip_punctuation(entity.label)}: {entity.detail}"
    return capitalise(_ensure_punctuation(text))


def _get_nodes_of_type(entities: Iterable[Entity], type_: EntityType) -> list[Entity]:
    """Get all entities whose `.type` matches `type_` as a new list."""
    return [e for e in entities if e.type is type_]


class PaperSection(Immutable):
    """Section of a full paper with its heading and content text."""

    heading: str
    text: str


class Paper(Record):
    """PeerRead paper with only currently useful fields."""

    title: str
    abstract: str
    reviews: Sequence[pr.PaperReview]
    authors: Sequence[str]
    sections: Sequence[PaperSection]
    rationale: str
    rating: int

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)


class ReviewEvaluation(Immutable):
    """Peer review with its original rating and predicted rating from GPT."""

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

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @computed_field
    @property
    def predicted_label(self) -> int | None:
        """Convert predicted rating rating to binary label."""
        if self.predicted_rating is None:
            return None
        return int(self.predicted_rating >= 3)


class PaperWithReviewEval(Record):
    """PeerRead paper with predicted reviews."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    reviews: Annotated[
        Sequence[ReviewEvaluation], Field(description="Feedback from a reviewer")
    ]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    references: Annotated[
        Sequence[pr.PaperReference],
        Field(description="References made in the paper"),
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]
    review: Annotated[ReviewEvaluation, Field(description="Main review for the paper")]
    rationale: Annotated[str, Field(description="Rationale for the main review")]
    rating: Annotated[int, Field(description="Rating (1-5) for the main review")]

    @computed_field
    @property
    def label(self) -> int:
        """Convert rating to binary label."""
        return int(self.rating >= 3)

    @property
    @override
    def id(self) -> str:
        return hashstr(self.title + self.abstract)

    def main_text(self) -> str:
        """Join all paper sections to form the main text."""
        return pr.clean_maintext("\n".join(s.text for s in self.sections))

    def __str__(self) -> str:
        """Display title, abstract, rating scores and count of words in main text."""
        main_text_words_num = len(self.main_text().split())
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Main text: {main_text_words_num} words.\n"
            f"Ratings: {[r.rating for r in self.reviews]}\n"
        )


class Prompt(Immutable):
    """Prompt used in a GPT API request."""

    system: str
    user: str


class PromptResult[T](Immutable):
    """Wrapper around a GPT API response with the full prompt that generated it."""

    item: T
    prompt: Prompt

    @classmethod
    def unwrap[U](cls, data: Iterable[PromptResult[U]]) -> list[U]:
        """Transform an iterable of wrapped items into a list of the internal elements."""
        return [x.item for x in data]

    def map[U](self, fn: Callable[[T], U]) -> PromptResult[U]:
        """Apply function to wrapped `item`."""
        return PromptResult(item=fn(self.item), prompt=self.prompt)


class PaperTermRelation(Immutable):
    """Represents a directed 'used for' relation between two scientific terms.

    Relations are always (head, used-for, tail).
    """

    head: Annotated[str, Field(description="Head term of the relation.")]
    tail: Annotated[str, Field(description="Tail term of the relation.")]


class PaperTerms(Immutable):
    """Structured output for scientific term extraction."""

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


class PaperType(StrEnum):
    """Whether the paper came from the S2 API or PeerRead dataset."""

    S2 = "s2"
    PeerRead = "peerread"

    def get_type(self) -> type[PaperToAnnotate]:
        """Returns concrete model type for the paper."""
        match self:
            case self.S2:
                return s2.Paper
            case self.PeerRead:
                return s2.PaperWithS2Refs


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


class PeerReadAnnotated(Record, PaperProxy[s2.PaperWithS2Refs]):
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


class PaperWithRelatedSummary(Record, PaperProxy[PeerReadAnnotated]):
    """PeerRead paper with its related papers formatted as prompt input."""

    paper: PeerReadAnnotated
    related: Sequence[PaperRelatedSummarised]

    @property
    def id(self) -> str:
        """Identify graph result as the underlying paper's ID."""
        return self.paper.id

    @property
    def terms(self) -> PaperTerms:
        """Terms extracted from the main paper."""
        return self.paper.terms

    @property
    def background(self) -> str:
        """Background text from the main paper annotation."""
        return self.paper.background

    @property
    def target(self) -> str:
        """Target text from the main paper annotation."""
        return self.paper.target


class RelatedPaperSource(StrEnum):
    """Denote where the related paper came from."""

    CITATIONS = "citations"
    SEMANTIC = "semantic"


class PaperRelatedSummarised(Record):
    """PETER-related paper with summary."""

    summary: str

    paper_id: str
    title: str
    abstract: str
    score: float
    polarity: pr.ContextPolarity
    source: RelatedPaperSource

    @property
    def id(self) -> str:
        """Identify the summary by its underlying paper ID."""
        return self.paper_id

    @classmethod
    def from_related(cls, related: rp.PaperRelated, summary: str) -> Self:
        """PETER-related paper with generated summary."""
        return cls(
            summary=summary,
            paper_id=related.paper_id,
            title=related.title,
            abstract=related.abstract,
            score=related.score,
            polarity=pr.ContextPolarity(related.polarity),
            source=RelatedPaperSource(related.source),
        )


type PaperACUInput = s2.Paper | s2.PaperWithS2Refs
"""Type of input paper, either from S2 or PeerRead/ORC."""


class PaperWithACUs[T: PaperACUInput](Record):
    """Paper (S2 or PeerRead) with extract atomic content units (ACUs)."""

    paper: T
    acus: Sequence[str]
    salient_acus: Sequence[str]
    summary: str

    @classmethod
    def from_(
        cls,
        paper: T,
        acus: Sequence[str],
        salient: Sequence[str],
        summary: str,
    ) -> Self:
        """New paper with extracted ACUs."""
        return cls(paper=paper, acus=acus, salient_acus=salient, summary=summary)

    @property
    @override
    def id(self) -> str:
        return self.paper.id


class PaperACUType(StrEnum):
    """Whether the paper came from the S2 API or PeerRead dataset."""

    S2 = "s2"
    PeerRead = "peerread"

    def get_type(self) -> type[PaperACUInput]:
        """Returns concrete model type for the paper."""
        match self:
            case self.S2:
                return s2.Paper
            case self.PeerRead:
                return s2.PaperWithS2Refs


class PeerPaperWithACUs(Record):
    """PeerRead Paper with extract atomic content units (ACUs)."""

    paper: pr.Paper
    acus: Sequence[str]
    salient_acus: Sequence[str]
    summary: str

    @classmethod
    def from_(
        cls,
        paper: pr.Paper,
        acus: Sequence[str],
        salient: Sequence[str],
        summary: str,
    ) -> Self:
        """New paper with extracted ACUs."""
        return cls(paper=paper, acus=acus, salient_acus=salient, summary=summary)

    @property
    @override
    def id(self) -> str:
        return self.paper.id
