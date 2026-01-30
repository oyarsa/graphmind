"""Server-Sent Events (SSE) utilities for FastAPI streaming responses."""

import asyncio
import contextlib
import json
import logging
import random
import string
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from enum import Enum
from typing import Any

import rich
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from paper import single_paper
from paper.backend.rate_limiter import RateLimiter
from paper.util import atimer
from paper.util.request_context import set_request_context

logger = logging.getLogger(__name__)


def _generate_request_id(length: int = 8) -> str:
    """Generates a request ID with `length` characters (ASCII letters and digits)."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


class _StreamStatus(Enum):
    """Status sentinel tokens for the stream."""

    DONE = 0
    """Stream progressing is done. Signals the processing loop to end."""


def create_streaming_response[T: BaseModel](
    rate_limiter: RateLimiter,
    rate_limit: str,
    request: Request,
    evaluation_func: Callable[[single_paper.ProgressCallback], Awaitable[T]],
    name: str,
    pulse_timeout_s: int = 15,
) -> StreamingResponse:
    """Create a StreamingResponse for Server-Sent Events with common streaming logic.

    Args:
        rate_limiter: RateLimiter instance to check rate limits.
        rate_limit: Rate limit string (e.g. '10/minute').
        request: FastAPI request object for rate limiting.
        evaluation_func: Async function that performs the evaluation, taking a progress
            callback.
        name: Name of the task (e.g. 'evaluation').
        pulse_timeout_s: Timeout for keep-alive messages in seconds.

    Returns:
        StreamingResponse configured for SSE.
    """
    # Manual rate limiting check to return SSE error event instead of HTTP exception
    if not rate_limiter.check_rate_limit(request, rate_limit):
        return _new_streaming_response(_rate_limit_error_stream())

    request_id = _generate_request_id()
    set_request_context(request_id)

    async def generate_events() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for progress updates and final result.

        Yields:
            Stringified events.
        """
        queue: asyncio.Queue[str | _StreamStatus] = asyncio.Queue()

        async def progress_cb(msg: str) -> None:
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
                await queue.put(_StreamStatus.DONE)

        # Kick things off
        yield _sse_event("connected", {"message": f"Starting {name}..."})
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

                if msg is _StreamStatus.DONE:
                    break

                yield _sse_event("progress", {"message": msg})

            # Task is done; surface its result or its error
            if exc := task.exception():
                yield _sse_event("error", {"message": str(exc)})
                logger.error(f"Evaluation failed: {exc}")
                return
            else:
                result = task.result()
                logger.info(f"{name} completed.")
                yield _sse_event("complete", {"result": result.model_dump()})

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Client disconnected - cancelling worker")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    return _new_streaming_response(generate_events())


def _sse_event(event: str | None, data: Any) -> str:
    """Format an SSE frame.

    Adding an `event:` field lets the client register addEventListener('progress', …) if
    it wants to.
    """
    payload = json.dumps(data)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def _rate_limit_error_stream() -> Generator[str, None, None]:
    """Generate SSE error event for rate limiting.

    Yields:
        Stringified error event with "too many requests" message.
    """
    yield _sse_event(
        "error",
        {"message": "Too many requests. Please wait a minute before trying again."},
    )


def _new_streaming_response(
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
