"""Rate limiter for the OpenAI API. Works for both request and token limits."""
# openlimit: Maximize your usage of OpenAI models without hitting rate limits

# Copyright (C) 2024 shobrook
#
# This file is part of openlimit.
#
# openlimit is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# openlimit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with openlimit. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------------
# Modifications:
# - 2024-10-25: oyarsa - Move only what I use to a single file and add type hints.
# ---------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

import tiktoken

_GPT_TOKENISER = tiktoken.get_encoding("cl100k_base")


class Message(TypedDict):
    role: str
    content: str
    name: str | None


class Bucket:
    def __init__(self, rate_limit: float, bucket_size_in_seconds: float) -> None:
        self._bucket_size_in_seconds = bucket_size_in_seconds
        self._rate_per_sec: float = rate_limit / 60
        self._capacity: float = self._rate_per_sec * self._bucket_size_in_seconds
        self._last_checked: float = time.time()

    def get_capacity(self, current_time: float) -> float:
        time_passed = current_time - self._last_checked
        return min(
            self._rate_per_sec * self._bucket_size_in_seconds,
            self._capacity + time_passed * self._rate_per_sec,
        )

    def set_capacity(self, new_capacity: float, current_time: float) -> None:
        self._last_checked = current_time
        self._capacity = new_capacity


class Buckets:
    def __init__(self, buckets: list[Bucket]) -> None:
        self.buckets = buckets

    def _get_capacities(self, current_time: float) -> list[float]:
        return [
            bucket.get_capacity(current_time=current_time) for bucket in self.buckets
        ]

    def _set_capacities(self, new_capacities: list[float], current_time: float) -> None:
        for new_capacity, bucket in zip(new_capacities, self.buckets):
            bucket.set_capacity(new_capacity, current_time=current_time)

    def _has_capacity(self, amounts: list[float]) -> bool:
        current_time = time.time()
        new_capacities = self._get_capacities(current_time=current_time)

        has_capacity = all(
            amount <= capacity for amount, capacity in zip(amounts, new_capacities)
        )

        if has_capacity:
            new_capacities = [
                capacity - amount for capacity, amount in zip(new_capacities, amounts)
            ]

        self._set_capacities(new_capacities, current_time=current_time)
        return has_capacity

    def wait_for_capacity_sync(
        self, amounts: list[float], sleep_interval: float = 0.1
    ) -> None:
        while not self._has_capacity(amounts):
            time.sleep(sleep_interval)

    async def wait_for_capacity(
        self, amounts: list[float], sleep_interval: float = 0.1
    ) -> None:
        while not self._has_capacity(amounts):
            await asyncio.sleep(sleep_interval)


class ChatRateLimiter:
    """Rate limiter for chat API requests with both request and token limits."""

    def __init__(
        self,
        request_limit: float = 3500,
        token_limit: float = 90000,
        bucket_size_in_seconds: float = 1.0,
    ) -> None:
        self.request_limit = request_limit
        self.token_limit = token_limit
        self.sleep_interval = 1 / (self.request_limit / 60)

        self._buckets = Buckets(
            [
                Bucket(request_limit, bucket_size_in_seconds),
                Bucket(token_limit, bucket_size_in_seconds),
            ]
        )

    @staticmethod
    def _count_tokens(
        messages: list[Message], max_tokens: int = 15, n: int = 1, **_: Any
    ) -> int:
        """Calculate total tokens that will be consumed by a chat request."""

        num_tokens = n * max_tokens

        for message in messages:
            # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(_GPT_TOKENISER.encode(str(value)))
                if key == "name":  # If there's a name, the role is omitted
                    num_tokens -= 1  # Role is always required and always 1 token

        # Every reply is primed with <im_start>assistant
        num_tokens += 2

        return num_tokens

    async def wait_for_capacity(self, num_tokens: int) -> None:
        """Wait asynchronously until capacity is available."""
        await self._buckets.wait_for_capacity(
            amounts=[1, num_tokens], sleep_interval=self.sleep_interval
        )

    def wait_for_capacity_sync(self, num_tokens: int) -> None:
        """Wait synchronously until capacity is available."""
        self._buckets.wait_for_capacity_sync(
            amounts=[1, num_tokens], sleep_interval=self.sleep_interval
        )

    def limit(self, **kwargs: Any) -> ContextManager:
        """Create a context manager that enforces the rate limits."""
        num_tokens = self._count_tokens(**kwargs)
        return ContextManager(num_tokens, self)


class ContextManager:
    """Context manager for rate limiting."""

    def __init__(self, num_tokens: int, rate_limiter: ChatRateLimiter) -> None:
        self.num_tokens = num_tokens
        self.rate_limiter = rate_limiter

    def __enter__(self) -> None:
        self.rate_limiter.wait_for_capacity_sync(self.num_tokens)

    def __exit__(self, *exc: Any) -> bool:
        return False

    async def __aenter__(self) -> None:
        await self.rate_limiter.wait_for_capacity(self.num_tokens)

    async def __aexit__(self, *exc: Any) -> bool:
        return False
