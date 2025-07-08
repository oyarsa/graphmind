"""Rate limiter for FastPI endpoints."""

import functools
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Concatenate

from fastapi import HTTPException, Request


def get_remote_address(request: Request) -> str:
    """Extract client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
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
        count, period = rule.split("/")
        count = int(count)

        # Convert period to seconds
        period_seconds = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }[period]

        # Include endpoint path in the key to separate limits per endpoint
        key = f"{self.key_func(request)}:{request.url.path}"
        current_time = time.time()

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
