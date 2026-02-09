"""OpenAI-based text encoder for embeddings."""

import asyncio
import itertools
from collections.abc import Coroutine, Sequence
from contextlib import suppress
from typing import Any

import backoff
import numpy as np
import openai

from paper.types import Matrix, Vector
from paper.util import ensure_envvar
from paper.util import progress as prog
from paper.util.rate_limiter import ChatRateLimiter

EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Rate limits for embedding models: (requests_per_minute, tokens_per_minute)
# From OpenAI API headers for tier 5
EMBEDDING_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "text-embedding-3-small": (10_000, 10_000_000),
    "text-embedding-3-large": (10_000, 10_000_000),
    "text-embedding-ada-002": (10_000, 10_000_000),
}


def _get_embedding_rate_limiter(model: str) -> ChatRateLimiter:
    """Get the rate limiter for an embedding model.

    Args:
        model: Embedding model name.

    Returns:
        Rate limiter configured for the model's limits.

    Raises:
        ValueError: If the model is not supported.
    """
    if model not in EMBEDDING_RATE_LIMITS:
        raise ValueError(
            f"Unknown embedding model: {model}. Known: {list(EMBEDDING_RATE_LIMITS)}"
        )

    request_limit, token_limit = EMBEDDING_RATE_LIMITS[model]
    return ChatRateLimiter(request_limit=request_limit, token_limit=token_limit)


class OpenAIEncoder:
    """Async OpenAI-based text encoder."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """Create an OpenAI embedding encoder.

        Args:
            model: Embedding model to use. Must be one of the known models.

        Raises:
            ValueError: If the model is not supported.
        """
        if model not in EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. Known: {list(EMBEDDING_DIMENSIONS)}"
            )
        self._client = openai.AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))
        self._model = model
        self._rate_limiter = _get_embedding_rate_limiter(model)

    @classmethod
    def from_env(cls, model: str = "text-embedding-3-small") -> "OpenAIEncoder":
        """Create encoder from environment variables.

        Args:
            model: Embedding model to use.

        Returns:
            Configured OpenAIEncoder instance.

        Requires:
            OPENAI_API_KEY environment variable.
        """
        return cls(model=model)

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the configured model."""
        return EMBEDDING_DIMENSIONS[self._model]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    async def encode(self, text: str) -> Vector:
        """Encode a single text string as a vector.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector as a numpy array.
        """
        async with self._rate_limiter.limit(
            contents=text, max_tokens=0
        ) as update_usage:
            response = await self._client.embeddings.create(
                input=[text], model=self._model
            )
            if usage := response.usage:
                await update_usage(usage.total_tokens)
        return np.array(response.data[0].embedding, dtype=np.float32)

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    async def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Encode multiple texts as a matrix.

        Args:
            texts: Sequence of texts to encode.

        Returns:
            Embedding matrix where each row is an embedding vector.
        """
        # Join texts for token estimation (separator doesn't affect count much)
        combined = "\n".join(texts)
        async with self._rate_limiter.limit(
            contents=combined, max_tokens=0
        ) as update_usage:
            response = await self._client.embeddings.create(
                input=list(texts), model=self._model
            )
            if usage := response.usage:
                await update_usage(usage.total_tokens)
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    async def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> Matrix:
        """Split `texts` into batches and encode each separately.

        Args:
            texts: Strings to encode.
            batch_size: Number of items per batch.
            progress: If True, show a progress bar for the process.

        Returns:
            The encoded vectors stacked as a single matrix.
        """
        batches = list(itertools.batched(texts, n=batch_size))
        tasks = [self.encode_multi(batch) for batch in batches]

        if progress:
            results = list(await prog.gather(tasks, desc="Batch text encoding"))
        else:
            results = list(await asyncio.gather(*tasks))

        return np.vstack(results)


class OpenAIEncoderSync:
    """Synchronous wrapper around `OpenAIEncoder`.

    Uses a dedicated event loop and runs embedding coroutines on it.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._encoder = OpenAIEncoder(model=model)
        self._model = model
        self._loop = asyncio.new_event_loop()

    def _run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the internal event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._loop.run_until_complete(coro)
        raise RuntimeError(
            "OpenAIEncoderSync cannot be used from an async context. "
            "Use OpenAIEncoder instead."
        )

    @property
    def model_name(self) -> str:
        """Model name configured for embeddings."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the configured model."""
        return self._encoder.dimensions

    def encode(self, text: str) -> Vector:
        """Encode a single text string as a vector."""
        return self._run(self._encoder.encode(text))

    def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Encode multiple texts as a matrix."""
        return self._run(self._encoder.encode_multi(texts))

    def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> Matrix:
        """Encode texts in batches and stack into a single matrix."""
        return self._run(
            self._encoder.batch_encode(texts, batch_size, progress=progress)
        )

    def close(self) -> None:
        """Close the async client and the internal event loop."""
        if not self._loop.is_closed():
            self._loop.run_until_complete(self._encoder.close())
            self._loop.close()

    def __enter__(self) -> "OpenAIEncoderSync":
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the encoder on context exit."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup."""
        with suppress(Exception):
            self.close()
