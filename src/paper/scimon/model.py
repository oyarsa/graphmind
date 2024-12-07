"""Model types for SciMON graph construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PaperAnnotated(Protocol):
    """Paper with an ID, annotated terms and background."""

    @property
    def title(self) -> str:
        """Paper title."""
        ...

    @property
    def terms(self) -> PaperTerms:
        """Terms and relations extracted from paper."""
        ...

    @property
    def id(self) -> str:
        """Unique identifier for the paper."""
        ...

    @property
    def background(self) -> str:
        """Background information from the paper (its problem, task, etc.)."""
        ...

    def target_terms(self) -> list[str]:
        """Get target terms from the paper, i.e. unique tail nodes from the relations."""
        ...


class PaperTerms(Protocol):
    """Annotated relations between scientific terms in a paper."""

    @property
    def relations(self) -> Sequence[PaperTermRelation]:
        """Used-for relations between terms."""
        ...


class PaperTermRelation(Protocol):
    """Directed 'used for' relation between two scientific terms (head -> tail)."""

    @property
    def head(self) -> str:
        """Head of the relation."""
        ...

    @property
    def tail(self) -> str:
        """Head of the relation."""
        ...
