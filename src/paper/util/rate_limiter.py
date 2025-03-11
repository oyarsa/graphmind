"""Rate limiter for the OpenAI API using a sliding window."""

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import tiktoken
from openai.types.chat import ChatCompletion

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


class Message(TypedDict):
    """Message sent to the OpenAI API."""

    role: str
    content: str
    name: str | None


def _message_num_tokens(value: object) -> int:
    """Count the number of tokens in `value`, treated as a string.

    Uses the GPT-4 tokeniser. If it somehow fails (e.g. invalid special tokens), we
    approximate the number as 1.5 * number of words (split by whitespace).
    """
    valstr = str(value)
    try:
        return len(_TOKENIZER.encode(valstr))
    except Exception:
        return int(len(valstr.split()) * 1.5)


def _count_tokens(
    messages: Iterable[Message], max_tokens: int | None = None, n: int = 1, **_: Any
) -> int:
    """Calculate total tokens that will be consumed by a chat request.

    The arguments have the same names as the ones from the OpenAI `client` API parameters.

    Args:
        messages: Data sent to the API.
        max_tokens: Maximum number of tokens the output. If None, defaults to 50.
        n: How many outputs are generated.
    """
    if max_tokens is None:
        max_tokens = 50
    num_tokens = n * max_tokens

    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += _message_num_tokens(value)
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
            bucket_size_in_seconds: Size of the bucket sliding window

        When `bucket_size_in_seconds` is 60s/1m (the default), the limits represent
        RPM and TPM, as is normally shown by OpenAI.
        """
        self._request_limit = request_limit
        self._token_limit = token_limit
        self._bucket_size = bucket_size_in_seconds

        # Track requests with timestamps and token usage
        # request_id -> (timestamp, tokens, is_actual)
        self._requests: dict[int, tuple[float, int, bool]] = {}
        self._lock = asyncio.Lock()

    async def _get_current_usage(self) -> tuple[int, int]:
        """Get current request and token usage in the sliding window."""

        current_time = time.time()
        cutoff_time = current_time - self._bucket_size

        # Remove expired entries
        expired_ids = [
            req_id
            for req_id, (timestamp, _, _) in self._requests.items()
            if timestamp < cutoff_time
        ]
        for req_id in expired_ids:
            del self._requests[req_id]

        current_requests = len(self._requests)
        current_tokens = sum(tokens for _, tokens, _ in self._requests.values())

        return current_requests, current_tokens

    @asynccontextmanager
    async def limit(
        self, messages: Iterable[Message], **kwargs: Any
    ) -> AsyncGenerator[Callable[[int | ChatCompletion], Coroutine[None, None, None]]]:
        """Context manager for waiting until for capacity before allowing the API call.

        It takes the messages being given to the API call and any other parameters that
        are passed to it.

        The context manager yield an update function to update the token usage with the
        following arguments:
            actual_tokens: The count of tokens, or a ChatCompletion object. If the
            completion has a valid usage, use `completion.usage.total_tokens`.

        Before the request, we estimate the number of tokens that will be spent based
        on the `max_token` request parameter. Use the update function for more precise
        tracking.
        """
        estimated_tokens = _count_tokens(messages, **kwargs)
        request_id = id(object())  # Generate a unique ID

        # Wait until there's capacity
        while True:
            async with self._lock:
                current_requests, current_tokens = await self._get_current_usage()

                if (
                    current_requests < self._request_limit
                    and current_tokens + estimated_tokens <= self._token_limit
                ):
                    # Record estimated usage
                    self._requests[request_id] = (time.time(), estimated_tokens, False)
                    break

            # No capacity, wait and retry
            await asyncio.sleep(0.05)

        try:
            # Callback for updating with actual token usage
            async def update_with_actual_usage(
                actual_tokens: int | ChatCompletion,
            ) -> None:
                if isinstance(actual_tokens, ChatCompletion):
                    usage = actual_tokens.usage
                    if usage is None:
                        return
                    actual_tokens = usage.total_tokens

                async with self._lock:
                    if request_id in self._requests:
                        timestamp, _, _ = self._requests[request_id]
                        self._requests[request_id] = (timestamp, actual_tokens, True)

            yield update_with_actual_usage
        finally:
            # We keep the request entry regardless of whether it was updated
            pass
