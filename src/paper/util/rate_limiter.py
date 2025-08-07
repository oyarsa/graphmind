"""Rate limiter for the OpenAI API using a sliding window."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

import tiktoken
from google.genai.types import GenerateContentResponse  # type: ignore
from openai.types.chat import ChatCompletion

_TOKENIZER = tiktoken.get_encoding("o200k_base")


class Message(TypedDict):
    """Message sent to the OpenAI API."""

    role: str
    content: str


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
        self,
        messages: Iterable[Message] | None = None,
        contents: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Callable[
            [int | ChatCompletion | GenerateContentResponse],
            Coroutine[None, None, None],
        ]
    ]:
        """Context manager for waiting until for capacity before allowing the API call.

        It takes the messages being given to the API call and any other parameters that
        are passed to it.

        The context manager yield an update function to update the token usage with the
        following arguments:
            actual_tokens: The count of tokens, a `ChatCompletion` object (OpenAI) or
                a `GenerateContentResponse` (Gemini).
            If the completion has a valid usage object, use total tokens from it.

        Before the request, we estimate the number of tokens that will be spent based
        on the `max_token` request parameter. Use the update function for more precise
        tracking.

        Yields:
            Function to update rate limiter usage with the actual usage from the API
            return.
        """
        if messages:
            # Fine, we'll use that as-is
            pass
        elif contents:
            # Transform Gemini-style contents string to OpenAI-style message:
            messages = [
                {"role": "user", "content": contents},
            ]
        else:
            raise ValueError("Either messages or contents need to be passed to limiter")

        estimated_tokens = _count_tokens(messages, **kwargs)
        request_id = uuid.uuid4().int

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
                actual_tokens: int | ChatCompletion | GenerateContentResponse,
            ) -> None:
                if isinstance(actual_tokens, ChatCompletion):
                    usage = actual_tokens.usage
                    if usage is None:
                        return
                    actual_tokens = usage.total_tokens
                elif isinstance(actual_tokens, GenerateContentResponse):
                    usage = actual_tokens.usage_metadata
                    if usage is None or usage.total_token_count is None:
                        return
                    actual_tokens = usage.total_token_count

                async with self._lock:
                    if request_id in self._requests:
                        timestamp, _, _ = self._requests[request_id]
                        self._requests[request_id] = (timestamp, actual_tokens, True)

            yield update_with_actual_usage
        finally:
            # We keep the request entry regardless of whether it was updated
            pass


type Limiter = asyncio.Semaphore | AsyncLimiter


def get_limiter(
    max_concurrent_requests: int = 1,
    requests_per_second: float = 1,
    use_semaphore: bool | None = None,
) -> Limiter:
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


# The AsyncLimiter code is derived from aiolimiter.
# SPDX-License-Identifier: MIT
# Copyright (c) 2019 Martijn Pieters
# Licensed under the MIT license as detailed in
# https://github.com/mjpieters/aiolimiter/blob/caf8c27ad1646b9b9fed1ebd78ce1331a2553f65/LICENSE.txt


LIMITER_REUSED_ACROSS_LOOPS_WARNING = (
    "This AsyncLimiter instance is being reused across loops. Please create "
    "a new limiter per event loop as reuse can lead to undefined behaviour."
)

_warn_reuse = partial(
    warnings.warn,
    message=LIMITER_REUSED_ACROSS_LOOPS_WARNING,
    category=RuntimeWarning,
    skip_file_prefixes=(str(Path(__file__).parent),),
)


class AsyncLimiter(AbstractAsyncContextManager[None]):
    """A leaky bucket rate limiter.

    This is an :ref:`asynchronous context manager <async-context-managers>`;
    when used with :keyword:`async with`, entering the context acquires
    capacity::

        limiter = AsyncLimiter(10)
        for foo in bar:
            async with limiter:
                # process foo elements at 10 items per minute

    :param max_rate: Allow up to `max_rate` / `time_period` acquisitions before
       blocking.
    :param time_period: duration, in seconds, of the time period in which to
       limit the rate. Note that up to `max_rate` acquisitions are allowed
       within this time period in a burst.

    """

    __slots__ = (
        "_event_loop",
        "_last_check",
        "_level",
        "_next_count",
        "_rate_per_sec",
        "_waiters",
        "_waker_handle",
        "max_rate",
        "time_period",
    )

    max_rate: float  #: The configured `max_rate` value for this limiter.
    time_period: float  #: The configured `time_period` value for this limiter.

    def __init__(self, max_rate: float, time_period: float = 60) -> None:
        self.max_rate = max_rate
        self.time_period = time_period
        self._rate_per_sec = max_rate / time_period
        self._level = 0.0
        self._last_check = 0.0

        # timer until next waiter can resume
        self._waker_handle: asyncio.TimerHandle | None = None
        # min-heap with (amount requested, order, future) for waiting tasks
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        # counter used to order waiting tasks
        self._next_count = partial(next, count())

    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        self._event_loop: asyncio.AbstractEventLoop
        try:
            loop = self._event_loop
            if loop.is_closed():
                # limiter is being reused across loops; make a best-effort
                # attempt at recovery. Existing waiters are ditched, with
                # the assumption that they are no longer viable.
                loop = self._event_loop = asyncio.get_running_loop()
                self._waiters = [
                    (amt, cnt, fut)
                    for amt, cnt, fut in self._waiters
                    if fut.get_loop() == loop
                ]
                _warn_reuse()

        except AttributeError:
            loop = self._event_loop = asyncio.get_running_loop()
        return loop

    def _leak(self) -> None:
        """Drip out capacity from the bucket."""
        now = self._loop.time()
        if self._level:
            # drip out enough level for the elapsed time since
            # we last checked
            elapsed = now - self._last_check
            decrement = elapsed * self._rate_per_sec
            self._level = max(self._level - decrement, 0)
        self._last_check = now

    def has_capacity(self, amount: float = 1) -> bool:
        """Check if there is enough capacity remaining in the limiter.

        :param amount: How much capacity you need to be available.

        """
        self._leak()
        return self._level + amount <= self.max_rate

    async def acquire(self, amount: float = 1) -> None:
        """Acquire capacity in the limiter.

        If the limit has been reached, blocks until enough capacity has been
        freed before returning.

        :param amount: How much capacity you need to be available.
        :exception: Raises :exc:`ValueError` if `amount` is greater than
           :attr:`max_rate`.

        """
        if amount > self.max_rate:
            raise ValueError("Can't acquire more than the maximum capacity")

        loop = self._loop
        while not self.has_capacity(amount):
            # Add a future to the _waiters heapq to be notified when capacity
            # has come up. The future callback uses call_soon so other tasks
            # are checked *after* completing capacity acquisition in this task.
            fut = loop.create_future()
            fut.add_done_callback(partial(loop.call_soon, self._wake_next))
            heappush(self._waiters, (amount, self._next_count(), fut))
            self._wake_next()
            await fut

        self._level += amount
        # reset the waker to account for the new, lower level.
        self._wake_next()

    def _wake_next(self, *_args: object) -> None:
        """Wake the next waiting future or set a timer."""
        # clear timer and any cancelled futures at the top of the heap
        heap, handle, self._waker_handle = self._waiters, self._waker_handle, None
        if handle is not None:
            handle.cancel()
        while heap and heap[0][-1].done():
            heappop(heap)

        if not heap:
            # nothing left waiting
            return

        amount, _, fut = heap[0]
        self._leak()
        needed = amount - self.max_rate + self._level
        if needed <= 0:
            heappop(heap)
            fut.set_result(None)
            # fut.set_result triggers another _wake_next call
            return

        wake_next_at = self._last_check + (1 / self._rate_per_sec * needed)
        self._waker_handle = self._loop.call_at(wake_next_at, self._wake_next)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the limiter."""
        args = f"max_rate={self.max_rate!r}, time_period={self.time_period!r}"
        state = f"level: {self._level:f}, waiters: {len(self._waiters)}"
        if (handle := self._waker_handle) and not handle.cancelled():
            microseconds = int((handle.when() - self._loop.time()) * 10**6)
            if microseconds > 0:
                state += f", waking in {microseconds} \N{MICRO SIGN}s"
        return f"<AsyncLimiter({args}) at {id(self):#x} [{state}]>"

    async def __aenter__(self) -> None:
        """Enter the context manager and acquire capacity."""
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the context manager. Does nothing."""
