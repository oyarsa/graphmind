"""Graph type with all members and excerpts from the text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, override

from pydantic import Field

from paper.gpt.graph_types.base import GPTGraphBase
from paper.gpt.model import Entity, EntityType, Excerpt, Graph, Relationship
from paper.types import Immutable
from paper.util import at


class GPTExcerpt(GPTGraphBase):
    """Graph representing the paper with excerpts from the text."""

    title: Annotated[str, Field(description="Title of the paper.")]
    primary_area: Annotated[
        str,
        Field(
            description="The primary subject area of the paper picked from the ICLR list of"
            " topics."
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
            " target `methods`."
        ),
    ]
    methods: Annotated[
        Sequence[MethodEntity],
        Field(
            description="Methods used to verify the claims, with connections to target"
            " `experiments`"
        ),
    ]
    experiments: Annotated[
        Sequence[ExperimentEntity],
        Field(description="Experiments designed to put methods in practice."),
    ]

    @override
    def to_graph(self, title: str, abstract: str) -> Graph:
        """Build a real `Graph` from the entities and their relationships."""

        # Track seen labels to detect duplicates
        label_map: dict[tuple[str, EntityType], str] = {}
        labels_seen: set[str] = set()

        def entity(
            label: str,
            type: EntityType,
            detail: str | None = None,
            excerpts: Sequence[Excerpt] | None = None,
        ) -> Entity:
            if label in labels_seen:
                unique_label = f"{label} [{type.value}]"
            else:
                labels_seen.add(label)
                unique_label = label
            label_map[label, type] = unique_label
            return Entity(
                label=unique_label, type=type, detail=detail, excerpts=excerpts
            )

        entities = [
            entity(self.title, EntityType.TITLE),
            entity(self.primary_area, EntityType.PRIMARY_AREA),
            *(entity(kw, EntityType.KEYWORD) for kw in self.keywords),
            entity(self.tldr, EntityType.TLDR),
            *(
                entity(c.label, EntityType.CLAIM, c.detail, c.excerpts)
                for c in self.claims
            ),
            *(
                entity(m.label, EntityType.METHOD, m.detail, m.excerpts)
                for m in self.methods
            ),
            *(
                entity(x.label, EntityType.EXPERIMENT, x.detail, x.excerpts)
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
                    target=label_map[target.label, EntityType.METHOD],
                )
                for c in self.claims
                for midx in c.method_indices
                if (target := at(self.methods, midx, "claim->method", title))
            ),
            *(
                Relationship(
                    source=label_map[m.label, EntityType.METHOD],
                    target=label_map[target.label, EntityType.EXPERIMENT],
                )
                for m in self.methods
                for eidx in m.experiment_indices
                if (target := at(self.experiments, eidx, "method->exp", title))
            ),
        ]

        return Graph(
            title=title,
            abstract=abstract,
            entities=entities,
            relationships=relationships,
        )


type Excerpts = Annotated[
    Sequence[Excerpt],
    Field(
        description="Text excerpts extracted verbatim from the paper content about this"
        " entity."
    ),
]


class ClaimEntity(Immutable):
    """Entity representing a claim made in the paper."""

    label: Annotated[
        str, Field(description="Summary label of a claim made by the paper.")
    ]
    detail: Annotated[str, Field(description="Detail text about the claim.")]
    method_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices for the `methods` connected to this claim in the"
            " `methods` list. There must be at least one connected `method`."
        ),
    ]
    excerpts: Excerpts


class MethodEntity(Immutable):
    """Entity representing a method described in the paper to support the claims."""

    label: Annotated[
        str,
        Field(
            description="Summary label of a method used to validate claims from the"
            " paper."
        ),
    ]
    detail: Annotated[str, Field(description="Detail text about the method.")]
    index: Annotated[
        int, Field(description="Index for this method in the `methods` list")
    ]
    experiment_indices: Annotated[
        Sequence[int],
        Field(
            description="Indices for the `experiments` connected to this method in the "
            " `experiments` list. There must be at least one connected `experiment`."
        ),
    ]
    excerpts: Excerpts


class ExperimentEntity(Immutable):
    """Entity representing an experiment used to validate a method from the paper."""

    label: Annotated[
        str,
        Field(
            description="Summary label of an experiment used to validate the methods"
            " from the paper."
        ),
    ]
    detail: Annotated[str, Field(description="Detail text about the experiment.")]
    index: Annotated[
        int, Field(description="Index for this experiment in the `experiments` list")
    ]
    excerpts: Excerpts
