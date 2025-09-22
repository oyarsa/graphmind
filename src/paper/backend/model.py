"""Pydantic models for Paper Explorer data structures.

Defines all data models used throughout the application including papers,
relationships, search results, and API response types.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, NewType

from pydantic import BaseModel, ConfigDict, Field, computed_field

from paper import gpt
from paper.gpt.evaluate_paper import EvidenceItem
from paper.util import hashstr


class Model(BaseModel):
    """Base class for immutable models.

    Provides common configuration for all models including:
    - Frozen (immutable) instances
    - Population by name for field aliases
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)


PaperId = NewType("PaperId", str)


class Paper(Model):
    """Complete paper information and metadata.

    Represents a research paper with all associated metadata
    including bibliographic information and citation metrics.

    Attributes:
        id: Unique identifier for the paper.
        title: Paper title.
        year: Publication year.
        authors: List of author names.
        abstract: Paper abstract text.
        venue: Publication venue (journal or conference).
        citation_count: Number of times this paper has been cited.
        doi: Digital Object Identifier (optional).
        pdf_url: URL to PDF version (optional).
    """

    id: Annotated[PaperId, Field(description="Unique ID of the paper.")]
    title: Annotated[str, Field(description="Title of the paper.")]
    year: Annotated[int, Field(description="Year of publication.")]
    authors: Annotated[Sequence[str], Field(description="List of author names.")]
    abstract: Annotated[str, Field(description="Abstract text of the paper.")]
    venue: Annotated[
        str, Field(description="Publication venue (journal or conference).")
    ]
    citation_count: Annotated[int, Field(description="Number of citations received.")]
    doi: Annotated[
        str | None, Field(description="Digital Object Identifier if available.")
    ]
    pdf_url: Annotated[
        str | None, Field(description="URL to PDF version if available.")
    ]


class RelatedType(StrEnum):
    """Types of relationships between papers.

    - Citation: Direct citation relationship (source cites target). Directed.
    - Semantic: Content similarity based on abstracts. Undirected.

    Attributes:
        CITATION: One paper cites another.
        SEMANTIC: Papers have similar content or topics.
    """

    CITATION = "citation"
    SEMANTIC = "semantic"


class Related(Model):
    """Relationship between two papers.

    Represents either a citation relationship or semantic similarity
    between two papers with an associated similarity score.

    Attributes:
        source: ID of the source paper.
        target: ID of the target paper.
        type_: Type of relationship (citation or semantic).
        similarity: Similarity score between papers (0.0-1.0).
    """

    source: Annotated[PaperId, Field(description="ID of the source paper.")]
    target: Annotated[PaperId, Field(description="ID of the target paper.")]
    type_: Annotated[
        RelatedType,
        Field(alias="type", description="Type of relationship between papers."),
    ]
    similarity: Annotated[float, Field(description="Similarity score between papers.")]


class PaperSearchResult(Model):
    """Simplified paper information for search results.

    Contains essential paper information with relevance scoring
    for search result display.

    Attributes:
        id: Unique paper identifier.
        title: Paper title.
        year: Publication year.
        authors: List of author names.
        relevance: Search relevance score.
    """

    id: Annotated[PaperId, Field(description="Unique ID of the paper.")]
    title: Annotated[str, Field(description="Title of the paper.")]
    year: Annotated[int, Field(description="Year of publication.")]
    authors: Annotated[Sequence[str], Field(description="List of author names.")]
    relevance: Annotated[float, Field(description="Search relevance score.")]


class SearchResult(Model):
    """Container for paper search results.

    Attributes:
        query: Original search query.
        results: List of matching papers.
        total: Total number of results returned.
    """

    query: Annotated[str, Field(description="Search query string.")]
    results: Annotated[
        Sequence[PaperSearchResult], Field(description="List of matching papers.")
    ]
    total: Annotated[int, Field(description="Total number of matching papers.")]


class PaperNeighbour(Paper):
    """Paper with relationship information.

    Extends Paper with additional fields for relationship context,
    used when returning papers related to a central paper.

    Attributes:
        similarity: Similarity score to the central paper.
        type_: Type of relationship (citation or semantic).
    """

    similarity: Annotated[
        float, Field(description="Similarity score to the source paper.")
    ]
    type_: Annotated[RelatedType, Field(alias="type", description="Relation type")]


class RelatedPaperNeighbourhood(Model):
    """Collection of papers related to a central paper.

    Represents the 'neighbourhood' of papers around a central paper,
    including relationship information and metadata.

    Attributes:
        paper_id: ID of the central paper.
        neighbours: List of related papers with similarity scores.
        total_papers: Total number of related papers returned.
    """

    paper_id: Annotated[PaperId, Field(description="ID of the central paper.")]
    neighbours: Annotated[
        Sequence[PaperNeighbour],
        Field(description="List of related neighbouring papers."),
    ]
    total_papers: Annotated[
        int, Field(description="Total number of papers in the database.")
    ]


class HealthCheck(Model):
    """Health check response model."""

    status: Annotated[str, Field(description="Health status indicator.")]
    timestamp: Annotated[str, Field(description="ISO timestamp of the check.")]
    version: Annotated[str, Field(description="Project version.")]


class AbstractEvaluationResponse(Model):
    """Response from abstract paper evaluation."""

    title: Annotated[str, Field(description="Paper title.")]
    abstract: Annotated[str, Field(description="Paper abstract.")]
    keywords: Annotated[Sequence[str], Field(description="Extracted keywords.")]
    background: Annotated[
        str, Field(description="Extracted background from the abstract.")
    ]
    target: Annotated[str, Field(description="Extracted target from the abstract.")]
    label: Annotated[int, Field(description="Binary novelty score.")]
    probability: Annotated[
        float | None, Field(description="Percentage chance of the paper being novel")
    ]
    paper_summary: Annotated[str, Field(description="Summary of paper contributions.")]
    supporting_evidence: Annotated[
        Sequence[EvidenceItem], Field(description="Evidence supporting novelty")
    ]
    contradictory_evidence: Annotated[
        Sequence[EvidenceItem], Field(description="Evidence contradicting novelty")
    ]
    conclusion: Annotated[str, Field(description="Final assessment.")]
    total_cost: Annotated[float, Field(description="Total GPT cost.")]
    related: Annotated[
        Sequence[gpt.PaperRelatedSummarised], Field(description="Related papers.")
    ]

    @computed_field
    @property
    def id(self) -> str:
        """Generate a unique ID for the evaluation based on title and abstract."""
        return hashstr(self.title + self.abstract)


# Configuration constants for paper evaluation
EVAL_PROMPT = "full-graph-structured"
GRAPH_PROMPT = "full"
DEMOS = "orc_4"
DEMO_PROMPT = "abstract"
MULTI_EVAL_PROMPT = "simple"
MULTI_SUMM_PROMPT = "simple"
MULTI_STRUCT_PROMPT = "structured_extraction"
BEST_OF_N = 5
