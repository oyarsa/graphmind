"""Tests for network router endpoints."""

import urllib.parse
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from paper.backend.api import app
from paper.backend.model import (
    Paper,
    RelatedPaperNeighbourhood,
    RelatedType,
    SearchResult,
)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Use TestClient as a context manager to handle lifespan events."""
    with TestClient(app) as client:
        yield client


def test_search_basic(client: TestClient) -> None:
    """Test basic search functionality."""
    response = client.get("/network/search?q=BERT")
    assert response.status_code == 200

    # Parse response as SearchResult model - will validate all fields
    result = SearchResult.model_validate(response.json())
    assert result.query == "BERT"
    assert result.total >= 0


def test_search_with_limit(client: TestClient) -> None:
    """Test search with custom limit."""
    response = client.get("/network/search?q=neural&limit=3")
    assert response.status_code == 200

    result = SearchResult.model_validate(response.json())
    assert result.query == "neural"
    assert len(result.results) <= 3


def test_search_missing_query(client: TestClient) -> None:
    """Test search without query parameter."""
    response = client.get("/network/search")
    assert response.status_code == 422


def test_get_paper_if_exists(client: TestClient) -> None:
    """Test retrieving a paper that exists."""
    # First search for a paper to get a valid ID
    search_response = client.get("/network/search?q=BERT&limit=1")
    if search_response.status_code == 200:
        search_result = SearchResult.model_validate(search_response.json())
        if search_result.results:
            paper_id = search_result.results[0].id

            # Now get the full paper details
            response = client.get(f"/network/papers/{paper_id}")
            assert response.status_code == 200

            # Parse as Paper model - validates all fields
            paper = Paper.model_validate(response.json())
            assert paper.id == paper_id


def test_get_paper_not_found(client: TestClient) -> None:
    """Test retrieving non-existent paper."""
    response = client.get("/network/papers/definitely-does-not-exist-12345")
    assert response.status_code == 404
    assert response.json()["detail"] == "Paper not found"


def test_related_papers_citation(client: TestClient) -> None:
    """Test getting citation-related papers."""
    # First get a valid paper ID
    search_response = client.get("/network/search?q=BERT&limit=1")
    if search_response.status_code == 200:
        search_result = SearchResult.model_validate(search_response.json())
        if search_result.results:
            paper_id = search_result.results[0].id

            # Get citation-related papers
            response = client.get(f"/network/related/{paper_id}?type=citation&limit=3")
            assert response.status_code == 200

            # Parse as RelatedPaperNeighbourhood model - validates all fields
            neighbourhood = RelatedPaperNeighbourhood.model_validate(response.json())
            assert neighbourhood.paper_id == paper_id
            assert len(neighbourhood.neighbours) <= 3

            # Check that all neighbours are citation type
            for neighbour in neighbourhood.neighbours:
                assert neighbour.type_ == RelatedType.CITATION


def test_related_papers_semantic(client: TestClient) -> None:
    """Test getting semantically-related papers."""
    # First get a valid paper ID
    search_response = client.get("/network/search?q=BERT&limit=1")
    if search_response.status_code == 200:
        search_result = SearchResult.model_validate(search_response.json())
        if search_result.results:
            paper_id = search_result.results[0].id

            # Get semantically-related papers
            response = client.get(f"/network/related/{paper_id}?type=semantic&limit=2")
            assert response.status_code == 200

            neighbourhood = RelatedPaperNeighbourhood.model_validate(response.json())
            assert neighbourhood.paper_id == paper_id
            assert len(neighbourhood.neighbours) <= 2

            # Check that all neighbours are semantic type
            for neighbour in neighbourhood.neighbours:
                assert neighbour.type_ == RelatedType.SEMANTIC


def test_related_missing_type(client: TestClient) -> None:
    """Test related endpoint without type parameter."""
    response = client.get("/network/related/some-paper-id")
    assert response.status_code == 422


def test_related_invalid_type(client: TestClient) -> None:
    """Test related endpoint with invalid type."""
    response = client.get("/network/related/some-paper-id?type=invalid")
    assert response.status_code == 422


def test_search_special_characters(client: TestClient) -> None:
    """Test search with special characters in query."""
    queries = [
        "machine learning",
        "COVID-19",
        "C++",
        "résumé",
        "naïve",
    ]

    for query in queries:
        # Properly encode the query parameter
        response = client.get(f"/network/search?q={urllib.parse.quote(query)}")
        assert response.status_code == 200
        result = SearchResult.model_validate(response.json())
        assert result.query == query


def test_limit_edge_cases(client: TestClient) -> None:
    """Test limit parameter edge cases."""
    # No limit should default to returning 5 entries
    response = client.get("/network/search?q=test")
    assert response.status_code == 200
    search = SearchResult.model_validate(response.json())
    assert len(search.results) <= 5

    # Limit above 100 should be an error
    response = client.get("/network/search?q=test&limit=10000")
    assert response.status_code == 422

    # Zero or negative limits should fail validation
    response = client.get("/network/search?q=test&limit=0")
    assert response.status_code == 422

    response = client.get("/network/search?q=test&limit=-1")
    assert response.status_code == 422


def test_response_consistency(client: TestClient) -> None:
    """Test that responses are consistent across multiple calls."""
    # Search for same query multiple times
    query = "machine learning"
    results: list[SearchResult] = []

    for _ in range(3):
        response = client.get(f"/network/search?q={query}&limit=5")
        assert response.status_code == 200
        results.append(SearchResult.model_validate(response.json()))

    # Results should be the same each time
    for i in range(1, len(results)):
        assert results[i].query == results[0].query
        assert results[i].total == results[0].total
        # Result order should be consistent
        if results[i].results and results[0].results:
            assert results[i].results[0].id == results[0].results[0].id
