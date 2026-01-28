"""OpenAI-based text encoder for embeddings."""

import asyncio
import itertools
from collections.abc import Sequence

import backoff
import numpy as np
import openai

from paper import embedding as emb
from paper.util import ensure_envvar
from paper.util import progress as prog

EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEncoder:
    """Async OpenAI-based text encoder."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        if model not in EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. Known: {list(EMBEDDING_DIMENSIONS)}"
            )
        self._client = openai.AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))
        self._model = model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the configured model."""
        return EMBEDDING_DIMENSIONS[self._model]

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    async def encode(self, text: str) -> emb.Vector:
        """Encode a single text string as a vector.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector as a numpy array.
        """
        response = await self._client.embeddings.create(input=[text], model=self._model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    async def encode_multi(self, texts: Sequence[str]) -> emb.Matrix:
        """Encode multiple texts as a matrix.

        Args:
            texts: Sequence of texts to encode.

        Returns:
            Embedding matrix where each row is an embedding vector.
        """
        response = await self._client.embeddings.create(
            input=list(texts), model=self._model
        )
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    async def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> emb.Matrix:
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
