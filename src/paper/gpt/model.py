import itertools
from collections import Counter, defaultdict
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from paper import hierarchical_graph


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


class Graph(BaseModel):
    model_config = ConfigDict(frozen=True)

    entities: Sequence[Entity]
    relationships: Sequence[Relationship]

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
                f"Validation: {validate_rules(self) or "valid"}",
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


def validate_rules(graph: Graph) -> str | None:
    """Check if graph rules hold. Returns error message if invalid, or None if valid.

    Rules:
    1. There must be exactly one node of types Title, Primary Area and TLDR.
    2. The Title node cannot have incoming edges.
    3. In the second level, TLDR, Primary Area and Keyword nodes can only have one
       incoming each, and it must be from Title.
    4. Primary Area, Keyword and Experiment nodes don't have outgoing edges.
    5. At later mid levels, nodes can only have incoming edges from the previous level
       and outgoing edges to the next level.
       The sequence in levels is:
        1. Title
        2. Title -> Primary Area, Keywords, TLDR
        3. TLDR -> Claims
        4. Claims -> Methods
        5. Methods -> Experiments
    6. At mid levels, in each node in a level must be connected to at least one node from
       the previous.
    7. There should be no cycles

    Note: this function doesn't throw an exception if the graph is invalid, it just
    returns the error message. The graph is allowed to be invalid, but it's useful to
    know why it's invalid.

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

    # Rule 1: Exactly one node from Title, Primary Area and TLDR
    singletons = [EntityType.TITLE, EntityType.PRIMARY_AREA, EntityType.TLDR]
    for node_type in singletons:
        nodes = _get_nodes_of_type(graph, node_type)
        if len(nodes) != 1:
            return f"Found {len(nodes)} '{node_type}' nodes. Should be exactly 1."

    # Rule 2: Title node cannot have incoming edges
    title = _get_nodes_of_type(graph, EntityType.TITLE)[0]
    if incoming[title.name]:
        return "Title node should not have any incoming edges."

    # Rule 3: TLDR, Primary Area and Keyword nodes only have incoming edges from Title
    level_2 = [EntityType.TLDR, EntityType.PRIMARY_AREA, EntityType.KEYWORD]
    for node_type in level_2:
        nodes = _get_nodes_of_type(graph, node_type)
        for node in nodes:
            inc_edges = incoming[node.name]
            if len(inc_edges) != 1:
                return (
                    f"Found {len(inc_edges)} incoming edges to node type '{node_type}'."
                    " Should be exactly 1."
                )

            inc_node = entities[inc_edges[0].source]
            if inc_node.type is not EntityType.TITLE:
                return f"Incoming edge to '{node_type}' is not Title, but '{inc_node.type}'."

    # Rule 4: Primary Area, Keyword and Experiment nodes don't have outgoing edges
    level_leaf = [EntityType.PRIMARY_AREA, EntityType.KEYWORD, EntityType.EXPERIMENT]
    for node_type in level_leaf:
        for node in _get_nodes_of_type(graph, node_type):
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
        for node in _get_nodes_of_type(graph, cur_type):
            for edge in outgoing[node.name]:
                type_ = entities[edge.target].type
                if type_ is not next_type:
                    return f"Found illegal outgoing edge from '{cur_type}' to '{type_}'"

    # Incoming edges
    for prev_type, cur_type in itertools.pairwise(level_order):
        for node in _get_nodes_of_type(graph, cur_type):
            for edge in incoming[node.name]:
                type_ = entities[edge.source].type
                if type_ is not prev_type:
                    return f"Found illegal incoming edge from '{type_}' to '{cur_type}'"

    # Rule 6: At mid levels, in each node in a level must be connected to at least one
    # node from the previous.
    for prev_type, cur_type in itertools.pairwise(level_order):
        for node in _get_nodes_of_type(graph, cur_type):
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

    # Rule 7: No cycles
    if graph_to_digraph(graph).has_cycle():
        return "Graph has cycles"

    return None


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
    DEFAULT = MEAN

    def is_approved(self, ratings: Sequence[int]) -> bool:
        match self:
            case RatingEvaluationStrategy.MEAN:
                mean = sum(ratings) / len(ratings)
                return mean >= RATING_APPROVAL_THRESHOLD
            case RatingEvaluationStrategy.MAJORITY:
                approvals = [r >= RATING_APPROVAL_THRESHOLD for r in ratings]
                return sum(approvals) >= len(approvals) / 2


class Paper(BaseModel):
    """ASAP-Review paper with only currently useful fields.

    Check the ASAP-Review dataset to see what else is available, and use
    paper.asap.extract and asap.filter to add them to this dataset.
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
