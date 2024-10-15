from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter


# Models from the ASAP files after exctraction (e.g. asap_filtered.json)
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


ASAPDatasetAdapter = TypeAdapter(list[Paper])


# Models after enrichment of references with data from the S2 API
class ReferenceWithAbstract(PaperReference):
    """ASAP reference with the added abstract and the original S2 title.

    `s2title` is the title in the S2 data for the best match. It can be used to match
    back to the original S2 file if desired.
    """

    abstract: str
    s2title: str


class PaperWithFullReference(BaseModel):
    """Paper from ASAP where the references contain their abstract."""

    model_config = ConfigDict(frozen=True)

    title: str
    abstract: str
    ratings: Sequence[int]
    sections: Sequence[PaperSection]
    approval: bool
    references: Sequence[ReferenceWithAbstract]


class S2Paper(BaseModel):
    """Paper from the S2 API.

    Attributes:
        title_query: the original title used to query the API.
        abstract: abstract text.

    NB: We got more data from the API, but this is what's relevant here. See also
    `paper_hypergraph.s2orc.query_s2`.
    """

    model_config = ConfigDict(frozen=True)

    title_query: str
    title: str
    abstract: str
