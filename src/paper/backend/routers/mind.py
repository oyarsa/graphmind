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
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Annotated, Any

import psutil
import rich
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paper import single_paper
from paper.backend.dependencies import EncoderDep, LimiterDep, LLMRegistryDep
from paper.backend.rate_limiter import RateLimiter
from paper.util import Timer, atimer, setup_logging

rate_limiter = RateLimiter()
router = APIRouter(prefix="/mind", tags=["mind"])
logger = logging.getLogger(__name__)
setup_logging()


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

    Args:
        request: Incoming request. Necessary for the rate limiter.
        q: Search query string for paper titles.
        limit: Maximum number of results to return (1-100).

    Returns:
        Search results containing matching papers from arXiv.
    """
    search_results = await single_paper.search_arxiv_papers(q, max_results=limit)
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


# Configuration constants for paper evaluation
EVAL_PROMPT = "full-graph-structured"
GRAPH_PROMPT = "full"
DEMOS = "orc_4"
DEMO_PROMPT = "abstract"


@router.get("/evaluate")
@rate_limiter.limit("5/minute")
async def evaluate(
    request: Request,
    limiter: LimiterDep,
    llm_registry: LLMRegistryDep,
    encoder: EncoderDep,
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

    Args:
        request: Incoming request. Necessary for the rate limiter.
        limiter: Rate limiter dependency for API calls.
        llm_registry: Registry of LLM clients.
        encoder: Text encoder for embeddings.
        id: arXiv ID of the paper to analyse.
        title: Title of the paper on arXiv.
        k_refs: Number of references to analyse (1-10).
        recommendations: Number of recommended papers to generate (5-50).
        related: Number of related papers to retrieve per type (1-10).
        llm_model: LLM model to use for analysis.
        filter_by_date: Filter recommended papers to only include those published
            before the main paper.

    Returns:
        StreamingResponse with Server-Sent Events containing progress updates and final
        result.
    """
    client = llm_registry.get_client(llm_model)

    @measure_memory
    async def go(
        callback: single_paper.ProgressCallback,
    ) -> single_paper.EvaluationResult:
        return await atimer(
            single_paper.process_paper_from_selection(
                client=client,
                title=title,
                arxiv_id=id,
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
        )

    def sse_event(event: str | None, data: Any) -> str:
        """Format an SSE frame.

        Adding an `event:` field lets the client register
        addEventListener('progress', …) if it wants to.
        """
        payload = json.dumps(data)
        prefix = f"event: {event}\n" if event else ""
        return f"{prefix}data: {payload}\n\n"

    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for progress updates and final result."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def progress_cb(msg: str) -> None:
            rich.print(f"[green]{msg}[/green]")
            await queue.put(msg)

        async def run_task() -> single_paper.EvaluationResult:
            try:
                return await go(progress_cb)
            finally:
                await queue.put(None)  # sentinel → stream is over

        # kick things off
        yield sse_event("connected", {"message": "Starting evaluation..."})
        task = asyncio.create_task(run_task())

        try:
            # stream until we see the sentinel
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    # 15-sec pulse keeps the pipe warm
                    yield ": keep-alive\n\n"
                    continue

                if msg is None:  # sentinel
                    break

                yield sse_event("progress", {"message": msg})

            # task is done; surface its result or its error
            if exc := task.exception():
                yield sse_event("error", {"message": str(exc)})
                logger.error(
                    f"Paper evaluation failed for '{title}' (arXiv:{id}): {exc}"
                )
                return
            else:
                evaluation_result = task.result()
                logger.info(f"Evaluation completed. Cost: {evaluation_result.cost}")

                yield sse_event("complete", {"result": evaluation_result.model_dump()})

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Client disconnected - cancelling worker")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
