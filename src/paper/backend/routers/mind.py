"""Mind API routes for paper analysis.

Provides endpoints for searching arXiv papers and performing comprehensive
paper evaluation using LLM-based analysis and recommendations.
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
import random
import string
import urllib.parse
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from enum import Enum, StrEnum
from typing import Annotated, Any

import psutil
import rich
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paper import single_paper
from paper.backend.dependencies import EncoderDep, LimiterDep, LLMRegistryDep
from paper.backend.model import (
    DEMO_PROMPT,
    DEMOS,
    EVAL_PROMPT,
    GRAPH_PROMPT,
    AbstractEvaluationResponse,
)
from paper.backend.rate_limiter import RateLimiter
from paper.util import Timer, atimer, setup_logging

rate_limiter = RateLimiter()
router = APIRouter(prefix="/mind", tags=["mind"])
logger = logging.getLogger(__name__)
setup_logging()

EVAL_RATE_LIMIT = "10/minute"


def generate_request_id(length: int = 8) -> str:
    """Generates a request ID with `length` characters (ASCII letters and digits)."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


class StreamStatus(Enum):
    """Status sentinel tokens for the stream."""

    DONE = 0
    """Stream progressing is done. Signals the processing loop to end."""


async def create_sse_streaming_response[T: BaseModel](
    request: Request,
    evaluation_func: Callable[[single_paper.ProgressCallback], Awaitable[T]],
    name: str,
    pulse_timeout_s: int = 15,
) -> StreamingResponse:
    """Create a StreamingResponse for Server-Sent Events with common streaming logic.

    Args:
        request: FastAPI request object for rate limiting.
        evaluation_func: Async function that performs the evaluation, taking a progress
            callback.
        name: Name of the task (e.g. 'evaluation').
        pulse_timeout_s: Timeout for keep-alive messages in seconds.

    Returns:
        StreamingResponse configured for SSE.
    """
    # Manual rate limiting check to return SSE error event instead of HTTP exception
    if not rate_limiter.check_rate_limit(request, EVAL_RATE_LIMIT):
        return new_streaming_response(rate_limit_error_stream())

    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for progress updates and final result."""
        queue: asyncio.Queue[str | StreamStatus] = asyncio.Queue()

        async def progress_cb(msg: str) -> None:
            request_id = generate_request_id()
            rich.print(f"[green]({request_id}) {name}: {msg}[/green]")
            await queue.put(msg)

        async def run_task() -> T:
            try:
                return await atimer(evaluation_func(progress_cb))
            except Exception as e:
                logger.exception(f"{name} failed")
                raise HTTPException(
                    status_code=500, detail=f"{name} failed: {e}"
                ) from e
            finally:
                await queue.put(StreamStatus.DONE)

        # Kick things off
        yield sse_event("connected", {"message": f"Starting {name}..."})
        task = asyncio.create_task(run_task())

        try:
            # stream until we see the sentinel
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=pulse_timeout_s)
                except TimeoutError:
                    # Periodic pulse keeps the pipe warm
                    yield ": keep-alive\n\n"
                    continue

                if msg is StreamStatus.DONE:
                    break

                yield sse_event("progress", {"message": msg})

            # Task is done; surface its result or its error
            if exc := task.exception():
                yield sse_event("error", {"message": str(exc)})
                logger.error(f"Evaluation failed: {exc}")
                return
            else:
                result = task.result()
                logger.info(f"{name} completed.")
                yield sse_event("complete", {"result": result.model_dump()})

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Client disconnected - cancelling worker")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    return new_streaming_response(generate_events())


def measure_memory[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Measures RAM used by func."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        process = psutil.Process(os.getpid())

        # Baseline memory
        mem_start = process.memory_info().rss / 1024 / 1024

        with Timer() as timer:
            result = await func(*args, **kwargs)

        # Final memory
        mem_end = process.memory_info().rss / 1024 / 1024

        logger.info(
            f"'{func.__name__}': "
            f"{mem_start:.1f} MB → {mem_end:.1f} MB (Δ {mem_end - mem_start:.1f} MB) | "
            f"{timer.seconds:.2f}s"
        )

        return result

    return wrapper


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

    return await create_sse_streaming_response(
        request=request, evaluation_func=go, name="evaluation"
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

    See OPTIONS /mind/evaluate-abstract from the result schema.
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
        )

    return await create_sse_streaming_response(
        request=request, evaluation_func=go, name="abstract evaluation"
    )


def sse_event(event: str | None, data: Any) -> str:
    """Format an SSE frame.

    Adding an `event:` field lets the client register
    addEventListener('progress', …) if it wants to.
    """
    payload = json.dumps(data)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def rate_limit_error_stream() -> Generator[str, None, None]:
    """Generate SSE error event for rate limiting."""
    yield sse_event(
        "error",
        {"message": "Too many requests. Please wait a minute before trying again."},
    )


def new_streaming_response(
    content_fn: Generator[str] | AsyncGenerator[str],
) -> StreamingResponse:
    """Create a StreamingResponse for Server-Sent Events (SSE)."""
    return StreamingResponse(
        content_fn,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
