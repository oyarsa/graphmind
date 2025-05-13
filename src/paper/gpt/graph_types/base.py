"""Abstract base class for GPT graph output types."""

from abc import ABC, abstractmethod

from paper.gpt.model import Graph
from paper.types import Immutable


class GPTGraphBase(Immutable, ABC):
    """Base class for all graph GPT output types."""

    @abstractmethod
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""
        ...
