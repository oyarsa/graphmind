"""Mind API routes for paper analysis.

Provides endpoints for searching arXiv papers and performing comprehensive
paper evaluation using LLM-based analysis and recommendations.
"""

import json
import logging
import secrets
import urllib.parse
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, cast

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
    ChatMessage,
    ChatPageType,
    ChatRequest,
    ChatResponse,
)
from paper.backend.rate_limiter import RateLimiter
from paper.util import atimer, measure_memory

rate_limiter = RateLimiter()
router = APIRouter(prefix="/mind", tags=["mind"])
logger = logging.getLogger(__name__)

EVAL_RATE_LIMIT = "10/minute"
CHAT_RATE_LIMIT = "30/minute"
MAX_CONTEXT_TEXT_CHARS = 12_000
MAX_PROMPT_CHARS = 18_000
MAX_JSON_DEPTH = 4
MAX_LIST_ITEMS = 12
MAX_KEY_CHARS = 64
MAX_SCALAR_CHARS = 800

CHAT_SYSTEM_PROMPT = (
    "You are a read-only paper discussion assistant for the GraphMind detail pages. "
    "Answer only questions about the provided paper context and conversation history. "
    "Do not invent missing facts. If information is unavailable in the context, say so "
    "clearly. You cannot perform UI actions, change settings, navigate pages, or mutate "
    "application state."
)


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to a bounded size with explicit suffix."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}… [truncated]"


def _sanitise_json(value: Any, depth: int = 0) -> Any:
    """Normalise semi-trusted JSON values into bounded prompt-safe data."""
    if depth >= MAX_JSON_DEPTH:
        return "[truncated-depth]"

    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return _truncate_text(value, MAX_SCALAR_CHARS)

    if isinstance(value, list):
        value_list = cast(list[Any], value)
        return [_sanitise_json(item, depth + 1) for item in value_list[:MAX_LIST_ITEMS]]

    if isinstance(value, dict):
        value_dict = cast(dict[str, Any], value)

        return {
            _truncate_text(key, MAX_KEY_CHARS): _sanitise_json(nested, depth + 1)
            for key, nested in list(value_dict.items())[:MAX_LIST_ITEMS]
        }

    return _truncate_text(str(value), MAX_SCALAR_CHARS)


def _normalise_page_context(
    page_type: ChatPageType,
    page_context: Mapping[str, Any],
) -> str:
    """Validate expected top-level shape and serialise bounded context JSON."""
    expected_keys = {
        ChatPageType.DETAIL: ("paper", "evaluation", "keywords", "related_papers"),
        ChatPageType.ABSTRACT_DETAIL: (
            "paper",
            "evaluation",
            "keywords",
            "related_papers",
        ),
    }[page_type]

    if not any(key in page_context for key in expected_keys):
        msg = (
            "page_context must include at least one expected key: "
            f"{', '.join(expected_keys)}"
        )
        raise HTTPException(status_code=422, detail=msg)

    normalised = {
        key: _sanitise_json(page_context[key])
        for key in expected_keys
        if key in page_context
    }

    payload = json.dumps(normalised, ensure_ascii=False)
    return _truncate_text(payload, MAX_CONTEXT_TEXT_CHARS)


def _build_chat_user_prompt(context_json: str, messages: Sequence[ChatMessage]) -> str:
    """Build a bounded user prompt from context and latest transcript turns."""
    prefix = "Page context (JSON):\n"
    transcript_header = "\n\nConversation transcript:\n"
    suffix = "\n\nRespond to the latest user message."

    # Preserve page context and keep the newest turns that fit the remaining budget.
    fixed_length = (
        len(prefix) + len(context_json) + len(transcript_header) + len(suffix)
    )
    transcript_budget = MAX_PROMPT_CHARS - fixed_length
    if transcript_budget <= 0:
        min_context_budget = max(
            0, MAX_PROMPT_CHARS - len(prefix) - len(transcript_header) - len(suffix)
        )
        context_json = _truncate_text(context_json, min_context_budget)
        fixed_length = (
            len(prefix) + len(context_json) + len(transcript_header) + len(suffix)
        )
        transcript_budget = max(0, MAX_PROMPT_CHARS - fixed_length)

    turns = [f"{message.role.value.title()}: {message.content}" for message in messages]
    selected_rev: list[str] = []
    used = 0
    for turn in reversed(turns):
        turn_len = len(turn) + (1 if selected_rev else 0)
        if used + turn_len > transcript_budget:
            break
        selected_rev.append(turn)
        used += turn_len

    selected = list(reversed(selected_rev))
    if not selected:
        latest_turn = turns[-1]
        if transcript_budget <= 0:
            selected = [""]
        else:
            selected = [_truncate_text(latest_turn, transcript_budget)]

    omission_marker = "[older turns omitted]"
    if len(selected) < len(turns):
        marker_len = len(omission_marker) + (1 if selected else 0)
        if len("\n".join(selected)) + marker_len <= transcript_budget:
            selected.insert(0, omission_marker)

    transcript = "\n".join(selected)
    return f"{prefix}{context_json}{transcript_header}{transcript}{suffix}"


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
    try:
        search_results = await atimer(
            single_paper.search_arxiv_papers_filtered(q, limit, check_latex=True)
        )
    except single_paper.ArxivRateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="Too many requests to arXiv. Please wait a moment and try again.",
        ) from e

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
    ] = True,
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
# Multi-perspective endpoint is currently disabled.
# To re-enable: remove the HTTPException raises below and uncomment the toggle
# in frontend/pages/search.html
@router.options(
    "/evaluate-multi", summary="Evaluate Multi-Perspective Schema Reference"
)
async def evaluation_multi_options() -> single_paper.EvaluationResultMulti:
    """This shows the schema of objects streamed by GET /evaluate-multi."""
    raise HTTPException(503, "Multi-perspective evaluation is temporarily disabled")


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
    ] = True,
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
    # Multi-perspective endpoint is currently disabled
    # To re-enable: remove this raise and uncomment the toggle in search.html
    raise HTTPException(503, "Multi-perspective evaluation is temporarily disabled")

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


@router.post("/chat", summary="Read-only paper chat")
@rate_limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    llm_registry: LLMRegistryDep,
) -> ChatResponse:
    """Generate a read-only chat response grounded in provided page context."""
    request_id = secrets.token_urlsafe(6)
    logger.info("(%s) Processing /mind/chat request", request_id)

    try:
        context_json = _normalise_page_context(
            page_type=chat_request.page_type,
            page_context=chat_request.page_context,
        )
        user_prompt = _build_chat_user_prompt(context_json, chat_request.messages)
        client = llm_registry.get_client(chat_request.llm_model)

        chat_result = await atimer(
            client.plain(
                system_prompt=CHAT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("(%s) /mind/chat failed", request_id)
        raise HTTPException(
            status_code=502,
            detail="Chat request failed. Please try again shortly.",
        ) from exc

    if not chat_result.result:
        logger.error("(%s) /mind/chat returned no content", request_id)
        raise HTTPException(
            status_code=502,
            detail="Chat request returned an empty response. Please retry.",
        )

    return ChatResponse(
        assistant_message=chat_result.result,
        cost=chat_result.cost,
    )
