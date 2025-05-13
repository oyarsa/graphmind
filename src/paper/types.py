"""Paper mixin type to automatically proxy paper properties."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class PaperSectionProtocol(Protocol):
    """Section of a paper with its heading and context text."""

    heading: str
    text: str


class PaperProtocol(Protocol):
    """Protocol defining the required interface for paper objects."""

    @property
    def id(self) -> str:
        """Paper ID."""
        ...

    @property
    def title(self) -> str:
        """Paper title."""
        ...

    @property
    def abstract(self) -> str:
        """Paper abstract."""
        ...

    @property
    def year(self) -> int | None:
        """Paper publication year."""
        ...

    @property
    def label(self) -> int:
        """Paper novelty label."""
        ...

    @property
    def rating(self) -> int:
        """Paper novelty rating."""
        ...

    @property
    def rationale(self) -> str:
        """Paper rationale."""
        ...

    @property
    def approval(self) -> bool | None:
        """Paper approval status."""
        ...

    @property
    def conference(self) -> str:
        """Paper conference."""
        ...

    @property
    def sections(self) -> Sequence[PaperSectionProtocol]:
        """Paper sections."""
        ...


class PaperProxy[P: PaperProtocol]:
    """Mixin class that provides proxy properties for paper."""

    paper: P

    @property
    def id(self) -> str:
        """ID of the underlying paper."""
        return self.paper.id

    @property
    def title(self) -> str:
        """Title of the underlying paper."""
        return self.paper.title

    @property
    def abstract(self) -> str:
        """Abstract of the underlying paper."""
        return self.paper.abstract

    @property
    def label(self) -> int:
        """Novelty label of the underlying paper."""
        return self.paper.label

    @property
    def rating(self) -> int:
        """Novelty rating of the underlying paper."""
        return self.paper.rating

    @property
    def rationale(self) -> str:
        """Rationale of the underlying paper."""
        return self.paper.rationale

    @property
    def approval(self) -> bool | None:
        """Approval of the underlying paper."""
        return self.paper.approval

    @property
    def conference(self) -> str:
        """Conference of the underlying paper."""
        return self.paper.conference

    @property
    def year(self) -> int | None:
        """Year of the underlying paper."""
        return self.paper.year

    @property
    def sections(self) -> Sequence[PaperSectionProtocol]:
        """Sections of the underlying paper."""
        return self.paper.sections


class Identifiable(Protocol):
    """Type with a unique ID."""

    @property
    def id(self) -> str:
        """Unique identification for the object."""
        ...


class Immutable(BaseModel):
    """Immutable BaseModel."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)
