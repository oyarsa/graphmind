"""Graph type with all members except for methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, override

from pydantic import BaseModel, Field

from paper.gpt.graph_types.base import GPTGraphBase
from paper.gpt.model import Entity, EntityType, Graph, Relationship
from paper.util import at


class GPTGraphNoMethods(GPTGraphBase):
    """Graph representing the paper."""

    title: Annotated[str, Field(description="Title of the paper.")]
    primary_area: Annotated[
        str,
        Field(
            description="The primary subject area of the paper picked from the ICLR"
            " list of topics."
        ),
    ]
    keywords: Annotated[
        Sequence[str],
        Field(description="Keywords that summarise the key aspects of the paper."),
    ]
    tldr: Annotated[str, Field(description="Sentence that summarises the paper.")]
    claims: Annotated[
        Sequence[ClaimEntity],
        Field(
            description="Main contributions the paper claims to make, with connections to"
            " target `experiments`."
        ),
    ]
    experiments: Annotated[
        Sequence[ExperimentEntity],
        Field(description="Experiments designed to supported the `claims`."),
    ]

    @override
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""

        # Track seen labels to detect duplicates
        label_map: dict[tuple[str, EntityType], str] = {}
        labels_seen: set[str] = set()

        def entity(label: str, type: EntityType, detail: str | None = None) -> Entity:
            if label in labels_seen:
                unique_label = f"{label} ({type.value})"
            else:
                labels_seen.add(label)
                unique_label = label
            label_map[label, type] = unique_label
            return Entity(label=unique_label, type=type, detail=detail)

        entities = [
            entity(self.title, EntityType.TITLE),
            entity(self.primary_area, EntityType.PRIMARY_AREA),
            *(entity(kw, EntityType.KEYWORD) for kw in self.keywords),
            entity(self.tldr, EntityType.TLDR),
            *(entity(c.label, EntityType.CLAIM, c.detail) for c in self.claims),
            *(
                entity(x.label, EntityType.EXPERIMENT, x.detail)
                for x in self.experiments
            ),
        ]

        relationships = [
            Relationship(
                source=label_map[self.title, EntityType.TITLE],
                target=label_map[self.primary_area, EntityType.PRIMARY_AREA],
            ),
            *(
                Relationship(
                    source=label_map[self.title, EntityType.TITLE],
                    target=label_map[kw, EntityType.KEYWORD],
                )
                for kw in self.keywords
            ),
            Relationship(
                source=label_map[self.title, EntityType.TITLE],
                target=label_map[self.tldr, EntityType.TLDR],
            ),
            *(
                Relationship(
                    source=label_map[self.tldr, EntityType.TLDR],
                    target=label_map[c.label, EntityType.CLAIM],
                )
                for c in self.claims
            ),
            *(
                Relationship(
                    source=label_map[c.label, EntityType.CLAIM],
                    target=label_map[target.label, EntityType.EXPERIMENT],
                )
                for c in self.claims
                for eidx in c.experiment_indices
                if (target := at(self.experiments, eidx, "claim->exp", title))
            ),
        ]

        return Graph(
            title=title,
            abstract=abstract,
            entities=entities,
            relationships=relationships,
        )


class ClaimEntity(BaseModel):
    """Entity representing a claim made in the paper."""

    label: Annotated[
        str, Field(description="Summary label of a claim made by the paper.")
    ]
    detail: Annotated[str, Field(description="Detail text about the claim.")]
    experiment_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices for the `experiments` connected to this claim in the"
            " `experiments` list. There must be at least one connected `experiment`."
        ),
    ]


class ExperimentEntity(BaseModel):
    """Entity representing an experiment used to support a claim from the paper."""

    label: Annotated[
        str,
        Field(
            description="Summary label of an experiment used to support claims from the"
            " paper."
        ),
    ]
    detail: Annotated[str, Field(description="Detail text about the experiment.")]
    index: Annotated[
        int, Field(description="Index for this experiment in the `experiments` list")
    ]
