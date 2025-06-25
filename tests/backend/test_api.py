"""Tests for general API functionality."""

import datetime as dt
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from paper.backend.api import app
from paper.backend.dependencies import ENABLE_NETWORK
from paper.backend.model import HealthCheck


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Use TestClient as a context manager to handle lifespan events."""
    with TestClient(app) as client:
        yield client


def test_health(client: TestClient) -> None:
    """Test health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = HealthCheck.model_validate(response.json())
    assert data.status == "ok"

    # Verify timestamp is valid ISO format
    timestamp = dt.datetime.fromisoformat(data.timestamp)
    assert isinstance(timestamp, dt.datetime)


def test_openapi_schema(client: TestClient) -> None:
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "Paper Explorer"
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/mind/search" in schema["paths"]
    assert "/mind/evaluate" in schema["paths"]

    if ENABLE_NETWORK:
        assert "/network/search" in schema["paths"]
        assert "/network/papers/{id}" in schema["paths"]
        assert "/network/related/{paper_id}" in schema["paths"]


def test_docs_available(client: TestClient) -> None:
    """Test that interactive docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
