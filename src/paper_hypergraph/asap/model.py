from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


class PaperReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    year: int
    authors: Sequence[str]
    contexts: Sequence[str]


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]
    approval: bool
    references: Sequence[PaperReference]


DatasetAdapter = TypeAdapter(list[Paper])
