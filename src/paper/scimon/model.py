"""Entities in the SciMON baseline reproduction."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

import paper.scimon.embedding as emb
from paper.scimon import citations, kg, semantic


class GraphData(BaseModel):
    model_config = ConfigDict(frozen=True)

    kg: kg.GraphData
    semantic: semantic.GraphData
    citations: citations.Graph
    encoder_model: str

    def to_graph(self, encoder: emb.Encoder) -> Graph:
        return Graph(
            kg=self.kg.to_graph(encoder),
            semantic=self.semantic.to_graph(encoder),
            citations=self.citations,
        )


@dataclass(frozen=True, kw_only=True)
class Graph:
    kg: kg.Graph
    semantic: semantic.Graph
    citations: citations.Graph
