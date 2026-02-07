"""Rate limiter for FastPI endpoints."""

import functools
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Concatenate

from fastapi import HTTPException, Request

_TRUST_PROXY_HEADERS_ENV = "XP_TRUST_PROXY_HEADERS"
_MAX_PERIOD_SECONDS = 86_400


def _trust_proxy_headers() -> bool:
    """Whether to trust proxy forwarding headers for remote IP extraction."""
    return os.getenv(_TRUST_PROXY_HEADERS_ENV, "0") == "1"


def get_remote_address(request: Request) -> str:
    """Extract client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded and _trust_proxy_headers():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


type EndpointFunc[**P, R] = Callable[Concatenate[Request, P], Awaitable[R]]


class RateLimiter:
    """Rate limiter for FastAPI endpoints.

    Example:
        limiter = Limiter()

        @app.get("/api/endpoint")
        @limiter.limit("5/minute")
        # The `request` parameter is mandatory.
        async def my_endpoint(request: Request):
            return {"message": "Hello"}
    """

    def __init__(self, key_func: Callable[[Request], str] | None = None) -> None:
        self.key_func = key_func or get_remote_address
        self._storage: dict[str, list[float]] = defaultdict(list)
        self._requests_seen = 0

    def limit[**P, R](
        self, rule: str
    ) -> Callable[[EndpointFunc[P, R]], EndpointFunc[P, R]]:
        """Decorator to apply rate limiting to an endpoint.

        Args:
            rule: Rate limit rule in format "count/period" (e.g., "5/minute").
        """

        def decorator(func: EndpointFunc[P, R]) -> EndpointFunc[P, R]:
            @functools.wraps(func)
            async def wrapper(request: Request, *args: P.args, **kwargs: P.kwargs) -> R:
                if not self.check_rate_limit(request, rule):
                    raise HTTPException(status_code=429, detail="Too Many Requests")
                return await func(request, *args, **kwargs)

            return wrapper

        return decorator

    def check_rate_limit(self, request: Request, rule: str) -> bool:
        """Check if a request would exceed the rate limit without raising an exception.

        Args:
            request: The FastAPI request object.
            rule: Rate limit rule in format "count/period" (e.g., "5/minute").

        Returns:
            True if the request is allowed, False if rate limited.
        """
        count, period_seconds = _parse_rule(rule)

        # Include endpoint path in the key to separate limits per endpoint
        key = f"{self.key_func(request)}:{request.url.path}"
        current_time = time.time()
        self._requests_seen += 1

        # Occasional global sweep to avoid unbounded stale keys.
        if self._requests_seen % 100 == 0:
            self._sweep_stale_keys(current_time)

        # Clean old entries
        self._storage[key] = [
            timestamp
            for timestamp in self._storage[key]
            if current_time - timestamp < period_seconds
        ]

        # Check if limit would be exceeded
        if len(self._storage[key]) >= count:
            return False

        # Record new request if allowed
        self._storage[key].append(current_time)
        return True

    def _sweep_stale_keys(self, current_time: float) -> None:
        """Remove keys that are completely stale for the longest supported window."""
        cutoff = current_time - _MAX_PERIOD_SECONDS
        stale_keys = [
            key
            for key, timestamps in self._storage.items()
            if all(timestamp < cutoff for timestamp in timestamps)
        ]
        for key in stale_keys:
            self._storage.pop(key, None)


def _parse_rule(rule: str) -> tuple[int, int]:
    """Parse `rule` and return `(count, period_seconds)`."""
    try:
        count_text, period_text = rule.split("/", maxsplit=1)
    except ValueError as e:
        raise ValueError(
            f"Invalid rate limit rule: {rule!r}. Expected format 'count/period'."
        ) from e

    try:
        count = int(count_text)
    except ValueError as e:
        raise ValueError(
            f"Invalid request count in rate limit rule: {count_text!r}."
        ) from e

    if count <= 0:
        raise ValueError(
            f"Invalid request count in rate limit rule: {count}. Must be positive."
        )

    period_seconds = {
        "second": 1,
        "seconds": 1,
        "minute": 60,
        "minutes": 60,
        "hour": 3600,
        "hours": 3600,
        "day": 86_400,
        "days": 86_400,
    }.get(period_text.strip().casefold())

    if period_seconds is None:
        raise ValueError(
            f"Invalid period in rate limit rule: {period_text!r}. "
            "Expected one of: second/minute/hour/day (singular or plural)."
        )

    return count, period_seconds
