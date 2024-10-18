from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


# Models from the ASAP files after exctraction (e.g. asap_filtered.json)
class PaperSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str = Field(description="Section heading")
    text: str = Field(description="Section full text")


class PaperReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the citation in the paper references")
    year: int = Field(description="Year of publication")
    authors: Sequence[str] = Field(description="Author names")
    contexts: Sequence[str] = Field(description="Citation contexts from this reference")


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[PaperReference] = Field(
        description="References made in the paper"
    )


ASAPDatasetAdapter = TypeAdapter(list[Paper])


# Models after enrichment of references with data from the S2 API
class ReferenceWithAbstract(PaperReference):
    """ASAP reference with the added abstract and the original S2 title.

    `s2title` is the title in the S2 data for the best match. It can be used to match
    back to the original S2 file if desired.
    """

    abstract: str = Field(description="Abstract text")
    s2title: str = Field(description="Title from the S2 data")


class PaperWithFullReference(BaseModel):
    """Paper from ASAP where the references contain their abstract."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[ReferenceWithAbstract] = Field(
        description="References made in the paper with their abstracts"
    )


class S2Paper(BaseModel):
    """Paper from the S2 API.

    Attributes:
        title_query: the original title used to query the API.
        abstract: abstract text.

    NB: We got more data from the API, but this is what's relevant here. See also
    `paper_hypergraph.s2orc.query_s2`.
    """

    model_config = ConfigDict(frozen=True)

    title_query: str = Field(description="Title used in the API query (from ASAP)")
    title: str = Field(description="Title from the S2 data")
    abstract: str = Field(description="Abstract text")
