import enum
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class EntityType(enum.StrEnum):
    TITLE = enum.auto()
    TLDR = enum.auto()
    PRIMARY_AREA = enum.auto()
    KEYWORD = enum.auto()
    CLAIM = enum.auto()
    METHOD = enum.auto()
    EXPERIMENT = enum.auto()


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
        node_type_counts = sorted(Counter(e.type for e in self.entities).items())

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
            ]
        )


def validate_rules(graph: Graph) -> str | None:
    """Check if graph rules hold. Returns error message if invalid, or None if valid.

    Rules:
    1. There must be exactly one Title node
    2. The Title node cannot have incoming edges
    3. The Title node can only have outgoing edges to Concepts
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

    return None


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


RATING_APPROVAL_THRESHOLD = 5
"""A rating is an approval if it's greater of equal than this."""


class RatingEvaluationStrategy(enum.StrEnum):
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
