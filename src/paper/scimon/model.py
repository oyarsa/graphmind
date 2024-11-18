from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Relation(BaseModel):
    """Represents a relation between two terms."""

    model_config = ConfigDict(frozen=True)

    head: str
    tail: str


class Terms(BaseModel):
    """Structured representation of key scientific terms from a paper."""

    model_config = ConfigDict(frozen=True)

    tasks: Sequence[str]
    methods: Sequence[str]
    metrics: Sequence[str]
    resources: Sequence[str]
    relations: Sequence[Relation]


class Paper(BaseModel):
    """Container for paper data."""

    model_config = ConfigDict(frozen=True)

    id: str
    terms: Terms
    context: str
    target: str
    abstract: str
