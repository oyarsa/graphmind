from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter


class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str
    text: str


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    introduction: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]
    approval: bool


DatasetAdapter = TypeAdapter(list[Paper])
