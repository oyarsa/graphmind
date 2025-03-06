"""Rate limiter for the OpenAI API using a sliding window."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import tiktoken

# Initialize the tokenizer - you may need to change this based on your model
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class Message(TypedDict):
    """Message sent to the OpenAI API."""

    role: str
    content: str
    name: str | None


def _count_tokens(
    messages: Iterable[Message], max_tokens: int = 50, n: int = 1, **_: Any
) -> int:
    """Calculate total tokens that will be consumed by a chat request.

    The arguments have the same names as the ones from the OpenAI `client` API parameters.

    Args:
        messages: Data sent to the API.
        max_tokens: Maximum number of tokens the output.
        n: How many outputs are generated.
    """
    num_tokens = n * max_tokens

    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(_TOKENIZER.encode(str(value)))
            if key == "name":  # If there's a name, the role is omitted
                num_tokens -= 1  # Role is always required and always 1 token

    # Every reply is primed with <im_start>assistant
    num_tokens += 2

    return num_tokens


class ChatRateLimiter:
    """Rate limiter for OpenAI API with both request and token limits."""

    def __init__(
        self, request_limit: int, token_limit: int, bucket_size_in_seconds: int = 60
    ) -> None:
        """Initialize the rate limiter.

        Args:
            request_limit: Maximum number of requests per bucket window
            token_limit: Maximum number of tokens per bucket window
            bucket_size_in_seconds: Size of the sliding window (default: 60s)
        """
        self.request_limit = request_limit
        self.token_limit = token_limit
        self.bucket_size = bucket_size_in_seconds

        # Sliding window tracking
        self.request_timestamps: deque[float] = deque()
        self.token_usage: deque[tuple[float, int]] = deque()

        # Lock to prevent race conditions
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def limit(
        self, messages: Iterable[Message], **_: Any
    ) -> AsyncGenerator[None, None]:
        """Context manager to rate limit API calls.

        Waits until there's capacity before allowing the API call.

        Call this with the same parameters as the actual OpenAI call. We'll use only
        `messages` and ignore the rest.

        Args:
            messages: Data sent to the API. We use this to estimate token usage.
        """
        num_tokens = _count_tokens(messages)

        await self._wait_for_capacity(num_tokens)

        try:
            yield
        finally:
            async with self._lock:
                self.request_timestamps.append(time.time())
                self.token_usage.append((time.time(), num_tokens))

    async def _wait_for_capacity(self, estimated_tokens: int) -> None:
        """Wait until there's capacity for another request.

        Args:
            estimated_tokens: Estimated token usage for this request
        """
        while True:
            current_time = time.time()
            cutoff_time = current_time - self.bucket_size

            async with self._lock:
                # Cleanup expired timestamps
                while (
                    self.request_timestamps and self.request_timestamps[0] < cutoff_time
                ):
                    self.request_timestamps.popleft()

                while self.token_usage and self.token_usage[0][0] < cutoff_time:
                    self.token_usage.popleft()

                # Check if we have capacity
                current_requests = len(self.request_timestamps)
                current_tokens = sum(tokens for _, tokens in self.token_usage)

                if (
                    current_requests < self.request_limit
                    and current_tokens + estimated_tokens <= self.token_limit
                ):
                    # We have capacity, exit the wait loop
                    break

            # No capacity, wait a bit and check again
            await asyncio.sleep(0.05)
