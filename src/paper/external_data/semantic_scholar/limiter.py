"""Limiter (rate limiter or semaphore) used for Semantic Scholar API requests."""

import asyncio
import os

from aiolimiter import AsyncLimiter


def get_limiter(
    max_concurrent_requests: int = 1,
    requests_per_second: float = 1,
    use_semaphore: bool | None = None,
) -> asyncio.Semaphore | AsyncLimiter:
    """Create some form of requests limiter based on the `USE_SEMAPHORE` env var.

    Args:
        max_concurrent_requests: When using a semaphore, the maximum number of requests
            (async tasks) that can execute simultaneously. The rest will wait until
            there's room available.
        requests_per_second: Number of requests per second that can be made.
        use_semaphore: Which one to use. If None, will check the USE_SEMAPHORE env var.
            If USE_SEMAPHORE is 1, use a Semaphore. Otherwise, use a rate limiter.

    Use Semaphore with small batches when you're not too worried about the rate limit,
    or Rate Limiiter when you want something more reliable.
    """
    if use_semaphore is None:
        use_semaphore = os.environ.get("USE_SEMAPHORE", "1") == "1"

    if use_semaphore:
        return asyncio.Semaphore(max_concurrent_requests)
    return AsyncLimiter(requests_per_second, 1)
