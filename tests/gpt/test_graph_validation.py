"""Tests for graph validation robustness with malformed graphs."""

from paper.gpt.model import Entity, EntityType, Graph, Relationship


def _entity(label: str, type_: EntityType) -> Entity:
    return Entity(label=label, type=type_)


def test_valid_status_handles_missing_title_without_crashing() -> None:
    """Validation should return errors instead of raising for missing title nodes."""
    graph = Graph(
        title="paper title",
        abstract="paper abstract",
        entities=[
            _entity("Main TLDR", EntityType.TLDR),
            _entity("Area", EntityType.PRIMARY_AREA),
        ],
        relationships=[],
    )

    errors = graph.valid_status_all
    assert any("Found 0 'title'" in error for error in errors)


def test_valid_status_handles_empty_incoming_edges_on_level2_nodes() -> None:
    """Level-2 nodes with no incoming edges should not cause indexing errors."""
    graph = Graph(
        title="paper title",
        abstract="paper abstract",
        entities=[
            _entity("Title", EntityType.TITLE),
            _entity("Main TLDR", EntityType.TLDR),
            _entity("Area", EntityType.PRIMARY_AREA),
        ],
        relationships=[],
    )

    errors = graph.valid_status_all
    assert any("Found 0 incoming edges to node type 'tldr'" in error for error in errors)


def test_valid_status_reports_edges_with_unknown_nodes() -> None:
    """Dangling relationships should be reported as validation errors."""
    graph = Graph(
        title="paper title",
        abstract="paper abstract",
        entities=[
            _entity("Title", EntityType.TITLE),
            _entity("Main TLDR", EntityType.TLDR),
            _entity("Area", EntityType.PRIMARY_AREA),
        ],
        relationships=[
            Relationship(source="Unknown node", target="Main TLDR"),
            Relationship(source="Title", target="Missing node"),
        ],
    )

    errors = graph.valid_status_all
    assert any("unknown source node" in error for error in errors)
    assert any("unknown target node" in error for error in errors)
