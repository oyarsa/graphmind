"""Shared HTTP helpers for Semantic Scholar API calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any

import aiohttp

type ResponseValidator = Callable[[aiohttp.ClientResponse], Awaitable[None] | None]

logger = logging.getLogger(__name__)


def _validator_noop(_: aiohttp.ClientResponse) -> None:
    return None


async def fetch_json_with_retries(
    session: aiohttp.ClientSession,
    *,
    params: Mapping[str, Any],
    url: str,
    max_tries: int,
    validate_response: ResponseValidator | None = None,
    limiter: AbstractAsyncContextManager[Any] | None = None,
) -> dict[str, Any]:
    """GET JSON from `url` with retries for client/network errors and timeouts.

    Uses exponential backoff with base 2.

    Args:
         session: aiohttp session to make the request.
         params: Get parameters for the request.
         url: URL for the request.
         max_tries: Maximum number of tries before giving up.
         validate_response: Runs on the returned request. Can be async or sync. Raise
            an exception if the response is invalid.
         limiter: If provided, acquires limiter capacity for each request attempt.

    Returns:
        JSON returned from the request as dict[str, Any].
    """
    if max_tries <= 0:
        raise ValueError(f"max_tries must be positive. Got: {max_tries}.")
    if validate_response is None:
        validate_response = _validator_noop
    if limiter is None:
        limiter = nullcontext()

    delay = 1.0
    for attempt in range(max_tries):
        try:
            async with limiter, session.get(url, params=params) as response:
                if (maybe_awaitable := validate_response(response)) is not None:
                    await maybe_awaitable

                return await response.json()
        except (aiohttp.ClientError, TimeoutError):
            if attempt + 1 == max_tries:
                logger.debug(
                    "Gave up fetching after %d tries. URL: %s. Params: %s",
                    max_tries,
                    url,
                    params,
                )
                raise
            await asyncio.sleep(delay)
            delay *= 2

    # Defensive fallback for type-checkers.
    raise RuntimeError("Unreachable: retry loop did not return or raise.")
