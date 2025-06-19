"""Network API routes for paper exploration.

Provides endpoints for searching papers, retrieving individual papers,
and exploring relationships between papers through citations and semantic similarity.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.params import Path

from paper.backend.db import DatabaseManager
from paper.backend.model import (
    Paper,
    PaperId,
    RelatedPaperNeighbourhood,
    RelatedType,
    SearchResult,
)

router = APIRouter(prefix="/network", tags=["network"])


def get_db(request: Request) -> DatabaseManager:
    """Dependency injection for the database manager.

    Args:
        request: FastAPI request object containing application state.

    Returns:
        DatabaseManager instance from application state.
    """
    return request.app.state.db


DbDep = Annotated[DatabaseManager, Depends(get_db)]


@router.get("/search")
async def search(
    db: DbDep,
    q: Annotated[str, Query(description="Query for paper title or abstract.")],
    limit: Annotated[
        int, Query(description="How many results to retrieve.", ge=1, le=100)
    ] = 5,
) -> SearchResult:
    """Search papers by title or abstract using full-text search.

    Performs PostgreSQL full-text search on paper titles and abstracts,
    returning results ranked by relevance score and citation count.

    Args:
        db: Database manager dependency.
        q: Search query string.
        limit: Maximum number of results (1-100).

    Returns:
        Search results with query, papers, and total count.
    """
    results = await db.search_papers(q, limit)
    return SearchResult(query=q, results=results, total=len(results))


@router.get("/papers/{id}")
async def paper(
    db: DbDep,
    id: Annotated[PaperId, Path(description="ID of the paper to retrieve.")],
) -> Paper:
    """Retrieve full information for a specific paper.

    Args:
        db: Database manager dependency.
        id: Unique identifier of the paper to retrieve.

    Returns:
        Complete paper information including metadata and content.

    Raises:
        HTTPException: 404 if paper is not found.
    """
    paper = await db.get_paper(id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.get("/related/{paper_id}")
async def related(
    db: DbDep,
    paper_id: Annotated[
        PaperId, Path(description="ID of the paper whose neighbours are retrieved.")
    ],
    type_: Annotated[
        RelatedType,
        Query(alias="type", description="What type of neighbours to retrieve."),
    ],
    limit: Annotated[
        int, Query(description="How many results to retrieve.", ge=1, le=100)
    ] = 5,
) -> RelatedPaperNeighbourhood:
    """Get related papers for a given paper ID.

    Returns papers related through citations or semantic similarity,
    including full paper information and similarity scores.

    For citation relationships: returns papers that cite the given paper.
    For semantic relationships: returns papers with similar content (bidirectional).

    Args:
        db: Database manager dependency.
        paper_id: ID of the central paper to find neighbours for.
        type_: Type of relationship (citation or semantic).
        limit: Maximum number of related papers to return (1-100).

    Returns:
        Neighbourhood containing the paper ID, related papers, and total count.
    """
    neighbours = await db.get_neighbours(paper_id, limit, type_)
    return RelatedPaperNeighbourhood(
        paper_id=paper_id, neighbours=neighbours, total_papers=len(neighbours)
    )
