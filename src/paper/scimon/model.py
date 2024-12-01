"""Model types for SciMON graph construction."""

from typing import Protocol

from paper.gpt.model import PaperTerms


class PaperAnnotated(Protocol):
    """Protocol for papers with an ID, annotated terms and background."""

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
