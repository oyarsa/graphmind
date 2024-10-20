from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RelationType(StrEnum):
    SUPPORT = "support"
    CONTRAST = "contrast"


class EntityType(StrEnum):
    TITLE = "title"
    CONCEPT = "concept"
    SENTENCE = "sentence"


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
