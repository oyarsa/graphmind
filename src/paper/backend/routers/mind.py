"""Mind API routes for paper analysis.

Provides endpoints for searching arXiv papers and performing comprehensive
paper evaluation using LLM-based analysis and recommendations.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

import rich
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from paper import evaluate

router = APIRouter(prefix="/mind", tags=["mind"])


def get_limiter(request: Request) -> evaluate.Limiter:
    """Dependency injection for the request rate limiter.

    Args:
        request: FastAPI request object containing application state.

    Returns:
        Rate limiter instance from application state.
    """
    return request.app.state.limiter


LimiterDep = Annotated[evaluate.Limiter, Depends(get_limiter)]


class PaperSearchItem(BaseModel):
    """Individual paper item from arXiv search results.

    Attributes:
        title: Paper title.
        arxiv_id: arXiv identifier.
        abstract: Paper abstract text.
        year: Publication year (may be None).
        authors: List of author names.
    """

    title: str
    arxiv_id: str
    abstract: str
    year: int | None
    authors: Sequence[str]


class PaperSearchResults(BaseModel):
    """Collection of paper search results from arXiv.

    Attributes:
        items: List of matching papers.
        query: Original search query.
        total: Total number of results returned.
    """

    items: Sequence[PaperSearchItem]
    query: str
    total: int


@router.get("/search")
async def search(
    q: Annotated[str, Query(description="Query for paper title on arXiv.")],
    limit: Annotated[
        int, Query(description="How many results to retrieve.", ge=1, le=100)
    ] = 5,
) -> PaperSearchResults:
    """Search papers on arXiv by title or abstract.

    Performs live search against the arXiv API to find relevant papers.

    Args:
        q: Search query string for paper titles.
        limit: Maximum number of results to return (1-100).

    Returns:
        Search results containing matching papers from arXiv.
    """
    search_results = await evaluate.search_arxiv_papers(q, max_results=limit)
    items = [
        PaperSearchItem(
            title=r.title,
            arxiv_id=evaluate.arxiv_id_from_url(r.entry_id),
            abstract=r.summary,
            year=r.published.year if r.published else None,
            authors=[a.name for a in r.authors],
        )
        for r in search_results
    ]
    return PaperSearchResults(items=items, query=q, total=len(items))


class LLMModel(StrEnum):
    """Available LLM models for paper analysis.

    Supported models for paper annotation, classification, and evaluation.
    """

    GPT4o = "gpt-4o"
    GPT4oMini = "gpt-4o-mini"
    Gemini2Flash = "gemini-2.0-flash"


# Configuration constants for paper evaluation
ENCODER_MODEL = evaluate.DEFAULT_SENTENCE_MODEL
EVAL_PROMPT = "full-graph-structured"
GRAPH_PROMPT = "full"
DEMOS = "orc_4"
DEMO_PROMPT = "abstract"


@router.get("/evaluate")
async def evaluate_(
    limiter: LimiterDep,
    id: Annotated[str, Query(description="ID of the paper to analyse.")],
    title: Annotated[str, Query(description="Title of the paper on arXiv.")],
    k_refs: Annotated[
        int, Query(description="How many references to use.", ge=1, le=10)
    ] = 2,
    recommendations: Annotated[
        int, Query(description="How many recommended papers to retrieve.", ge=5, le=50)
    ] = 30,
    related: Annotated[
        int,
        Query(
            description="How many related papers to retrieve, per type.", ge=1, le=10
        ),
    ] = 2,
    llm_model: Annotated[
        LLMModel, Query(description="LLM model to use.")
    ] = LLMModel.GPT4oMini,
    seed: Annotated[int, Query(description="Random seed.")] = 0,
) -> evaluate.EvaluationResult:
    """Perform comprehensive paper analysis and evaluation.

    Analyses an arXiv paper using LLM-based evaluation, including:
    - Paper classification and annotation
    - Reference analysis
    - Recommendation generation
    - Related paper discovery

    Args:
        limiter: Rate limiter dependency for API calls.
        id: arXiv ID of the paper to analyse.
        title: Title of the paper on arXiv.
        k_refs: Number of references to analyse (1-10).
        recommendations: Number of recommended papers to generate (5-50).
        related: Number of related papers to retrieve per type (1-10).
        llm_model: LLM model to use for analysis.
        seed: Random seed for reproducible results.

    Returns:
        Comprehensive evaluation result with analysis and recommendations.
    """

    def callback(msg: str) -> None:
        rich.print(f"[green]{msg}[/green]")

    return await evaluate.process_paper_from_selection(
        title=title,
        arxiv_id=id,
        top_k_refs=k_refs,
        num_recommendations=recommendations,
        num_related=related,
        llm_model=str(llm_model),
        encoder_model=ENCODER_MODEL,
        seed=seed,
        limiter=limiter,
        eval_prompt_key=EVAL_PROMPT,
        graph_prompt_key=GRAPH_PROMPT,
        demonstrations_key=DEMOS,
        demo_prompt_key=DEMO_PROMPT,
        callback=callback,
    )
