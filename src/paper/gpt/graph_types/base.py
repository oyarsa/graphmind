"""Abstract base class for GPT graph output types."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from paper.gpt.model import Graph


class GPTGraphBase(BaseModel, ABC):
    """Base class for all graph GPT output types."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""
        ...
