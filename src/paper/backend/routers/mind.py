"""Mind API routes for paper analysis.

Provides endpoints for searching arXiv papers and performing comprehensive
paper evaluation using LLM-based analysis and recommendations.
"""

import logging
import urllib.parse
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paper import single_paper
from paper.backend import sse
from paper.backend.dependencies import EncoderDep, LimiterDep, LLMRegistryDep
from paper.backend.model import (
    DEMO_PROMPT,
    DEMOS,
    EVAL_PROMPT,
    GRAPH_PROMPT,
    MULTI_EVAL_PROMPT,
    MULTI_STRUCT_PROMPT,
    MULTI_SUMM_PROMPT,
    AbstractEvaluationResponse,
)
from paper.backend.rate_limiter import RateLimiter
from paper.util import atimer, measure_memory

rate_limiter = RateLimiter()
router = APIRouter(prefix="/mind", tags=["mind"])
logger = logging.getLogger(__name__)

EVAL_RATE_LIMIT = "10/minute"


class PaperSearchItem(BaseModel):
    """Individual paper item from arXiv search results."""

    title: Annotated[str, Field(description="Paper title.")]
    arxiv_id: Annotated[str, Field(description="arXiv identifier.")]
    abstract: Annotated[str, Field(description="Paper abstract text.")]
    year: Annotated[int | None, Field(description="Publication year, if available.")]
    authors: Annotated[Sequence[str], Field(description="List of author names.")]


class PaperSearchResults(BaseModel):
    """Collection of paper search results from arXiv."""

    items: Annotated[Sequence[PaperSearchItem], Field(description="Items retrieved.")]
    query: Annotated[str, Field(description="Title used for the query.")]
    total: Annotated[int, Field(description="Number of items retrieved.")]


@router.get("/search")
@measure_memory
@rate_limiter.limit("1/second")
async def search(
    request: Request,
    q: Annotated[str, Query(description="Query for paper title on arXiv.")],
    limit: Annotated[
        int, Query(description="How many results to retrieve.", ge=1, le=100)
    ] = 10,
) -> PaperSearchResults:
    """Search papers on arXiv by title or abstract.

    Performs live search against the arXiv API to find relevant papers.
    Only returns papers that have LaTeX content available.
    """
    search_results = await atimer(
        single_paper.search_arxiv_papers_filtered(q, limit, check_latex=True)
    )

    if search_results is None:
        raise HTTPException(status_code=503, detail="arXiv API error")

    items = [
        PaperSearchItem(
            title=r.title,
            arxiv_id=single_paper.arxiv_id_from_url(r.entry_id),
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


# This endpoint, like `evaluation_partial_options` below, is necessary to add the
# output schema to the generate docs. Because SSE endpoints return string-typed
# responses, they can't document themselves.
@router.options("/evaluate", summary="Evaluate Schema Reference")
async def evaluation_options() -> single_paper.EvaluationResult:
    """This shows the schema of objects streamed by GET /evaluate."""
    raise HTTPException(501, "Use GET method for the actual SSE stream")


@router.get("/evaluate", summary="Evaluate (SSE)")
async def evaluate(
    request: Request,
    limiter: LimiterDep,
    llm_registry: LLMRegistryDep,
    encoder: EncoderDep,
    id: Annotated[str, Query(description="ID of the paper to analyse.")],
    title: Annotated[str, Query(description="Title of the paper on arXiv.")],
    k_refs: Annotated[
        int, Query(description="How many references to use.", ge=10, le=50)
    ] = 20,
    recommendations: Annotated[
        int, Query(description="How many recommended papers to retrieve.", ge=20, le=50)
    ] = 30,
    related: Annotated[
        int,
        Query(
            description="How many related papers to retrieve, per type.", ge=5, le=10
        ),
    ] = 5,
    llm_model: Annotated[
        LLMModel, Query(description="LLM model to use.")
    ] = LLMModel.Gemini2Flash,
    filter_by_date: Annotated[
        bool,
        Query(
            description="Filter recommended papers to only include those published"
            " before the main paper."
        ),
    ] = False,
) -> StreamingResponse:
    """Perform comprehensive paper analysis and evaluation with real-time progress.

    Analyses an arXiv paper using LLM-based evaluation, including:
    - Paper classification and annotation
    - Reference analysis
    - Recommendation generation
    - Related paper discovery

    Returns progress updates via Server-Sent Events (SSE), followed by the final result.

    See OPTIONS /mind/evaluate from the result schema.
    """
    client = llm_registry.get_client(llm_model)
    arxiv_id = urllib.parse.unquote_plus(id)

    @measure_memory
    async def go(
        callback: single_paper.ProgressCallback,
    ) -> single_paper.EvaluationResult:
        return await single_paper.process_paper_from_selection(
            client=client,
            title=title,
            arxiv_id=arxiv_id,
            encoder=encoder,
            top_k_refs=k_refs,
            num_recommendations=recommendations,
            num_related=related,
            limiter=limiter,
            eval_prompt_key=EVAL_PROMPT,
            graph_prompt_key=GRAPH_PROMPT,
            demonstrations_key=DEMOS,
            demo_prompt_key=DEMO_PROMPT,
            filter_by_date=filter_by_date,
            callback=callback,
        )

    return sse.create_streaming_response(
        rate_limiter=rate_limiter,
        rate_limit=EVAL_RATE_LIMIT,
        request=request,
        evaluation_func=go,
        name="evaluation",
    )


# See `evaluation_options` for why this exists.
@router.options("/evaluate-abstract", summary="Evaluate Abstract Schema Reference")
async def evaluation_abstract_options() -> AbstractEvaluationResponse:
    """This shows the schema of objects streamed by GET /evaluate-abstract."""
    raise HTTPException(501, "Use GET method for the actual SSE stream")


@router.get("/evaluate-abstract", summary="Evaluate Abstract (SSE)")
async def evaluate_abstract(
    request: Request,
    llm_registry: LLMRegistryDep,
    limiter: LimiterDep,
    encoder: EncoderDep,
    title: str,
    abstract: str,
    recommendations: Annotated[
        int, Query(description="Number of related papers to retrieve.", ge=20, le=50)
    ] = 20,
    llm_model: Annotated[
        LLMModel, Query(description="LLM model to use.")
    ] = LLMModel.Gemini2Flash,
    related: Annotated[
        int,
        Query(
            description="How many related papers to retrieve, per type.", ge=5, le=10
        ),
    ] = 5,
    year: Annotated[
        int | None,
        Query(
            description="Publication year of the main paper. If not provided, defaults"
            " to the current year. Related papers will be filtered to only include"
            " those published before this year."
        ),
    ] = None,
) -> StreamingResponse:
    """Evaluate paper novelty using only title and abstract with real-time progress.

    Performs simplified paper evaluation for unpublished papers, including:
    - Research context extraction from abstract
    - Related paper discovery via semantic search
    - Semantic similarity-based paper ranking
    - GPT-based structured novelty evaluation

    This endpoint is designed for unpublished papers where full content is not available.
    It uses Semantic Scholar search instead of citation analysis and provides faster
    evaluation with reduced accuracy compared to the full evaluation pipeline.

    Returns progress updates via Server-Sent Events (SSE), followed by the final result.

    See OPTIONS /mind/evaluate-abstract for the result schema.
    """
    client = llm_registry.get_client(llm_model)

    @measure_memory
    async def go(callback: single_paper.ProgressCallback) -> AbstractEvaluationResponse:
        """Run the abstract evaluation pipeline."""
        return await single_paper.abstract_evaluation(
            client=client,
            limiter=limiter,
            encoder=encoder,
            callback=callback,
            num_recommendations=recommendations,
            title=title,
            abstract=abstract,
            num_semantic=related,
            year=year,
        )

    return sse.create_streaming_response(
        rate_limiter=rate_limiter,
        rate_limit=EVAL_RATE_LIMIT,
        request=request,
        evaluation_func=go,
        name="abstract evaluation",
    )


# See `evaluation_options` for why this exists.
@router.options(
    "/evaluate-multi", summary="Evaluate Multi-Perspective Schema Reference"
)
async def evaluation_multi_options() -> single_paper.EvaluationResultMulti:
    """This shows the schema of objects streamed by GET /evaluate-multi."""
    raise HTTPException(501, "Use GET method for the actual SSE stream")


@router.get("/evaluate-multi", summary="Evaluate multi-perspectives (SSE)")
async def evaluate_multi(
    request: Request,
    limiter: LimiterDep,
    llm_registry: LLMRegistryDep,
    encoder: EncoderDep,
    id: Annotated[str, Query(description="ID of the paper to analyse.")],
    title: Annotated[str, Query(description="Title of the paper on arXiv.")],
    k_refs: Annotated[
        int, Query(description="How many references to use.", ge=10, le=50)
    ] = 20,
    recommendations: Annotated[
        int, Query(description="How many recommended papers to retrieve.", ge=20, le=50)
    ] = 30,
    related: Annotated[
        int,
        Query(
            description="How many related papers to retrieve, per type.", ge=5, le=10
        ),
    ] = 5,
    llm_model: Annotated[
        LLMModel, Query(description="LLM model to use.")
    ] = LLMModel.Gemini2Flash,
    filter_by_date: Annotated[
        bool,
        Query(
            description="Filter recommended papers to only include those published"
            " before the main paper."
        ),
    ] = False,
) -> StreamingResponse:
    """Perform comprehensive paper analysis and evaluation with real-time progress.

    Uses multi-perspective evaluation. Each perspective is evaluated separately, with
    combining them to obtain a final result.

    Analyses an arXiv paper using LLM-based evaluation, including:
    - Paper classification and annotation
    - Reference analysis
    - Recommendation generation
    - Related paper discovery

    Returns progress updates via Server-Sent Events (SSE), followed by the final result.

    See OPTIONS /mind/evaluate for the result schema.
    """
    client = llm_registry.get_client(llm_model)
    arxiv_id = urllib.parse.unquote_plus(id)

    @measure_memory
    async def go(
        callback: single_paper.ProgressCallback,
    ) -> single_paper.EvaluationResultMulti:
        return await single_paper.process_paper_from_selection_multi(
            client=client,
            title=title,
            arxiv_id=arxiv_id,
            encoder=encoder,
            top_k_refs=k_refs,
            num_recommendations=recommendations,
            num_related=related,
            limiter=limiter,
            eval_prompt_key=MULTI_EVAL_PROMPT,
            graph_prompt_key=GRAPH_PROMPT,
            summ_prompt_key=MULTI_SUMM_PROMPT,
            struct_prompt_key=MULTI_STRUCT_PROMPT,
            demonstrations_key=DEMOS,
            demo_prompt_key=DEMO_PROMPT,
            filter_by_date=filter_by_date,
            callback=callback,
        )

    return sse.create_streaming_response(
        rate_limiter=rate_limiter,
        rate_limit=EVAL_RATE_LIMIT,
        request=request,
        evaluation_func=go,
        name="evaluation",
    )
