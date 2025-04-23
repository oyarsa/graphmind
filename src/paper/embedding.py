"""Tools to generate embeddings from text using SentenceTransformers."""

import base64
import itertools
import logging
import math
import os
from collections.abc import Sequence
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

type Vector = npt.NDArray[np.float32]
type Matrix = npt.NDArray[np.float32]

logger = logging.getLogger(__name__)

DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"


class Encoder:
    """SentenceTransformer-based text to vector encoder.

    Supports both single-item query and multipe-items queries in parallel.
    """

    def __init__(
        self, model_name: str = DEFAULT_SENTENCE_MODEL, device: str | None = None
    ) -> None:
        # `sentence_transformers` has a bug where they don't clean up their semaphores
        # properly, so we suppress this.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(
            model_name, device=device or _get_best_device()
        )

    @property
    def dimensions(self) -> int | None:
        """Dimensions of the embeddings of the model."""
        return self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> Vector | Matrix:
        """Encode text string as a vector."""
        return cast(Vector, self._model.encode(text))  # type: ignore

    def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Encode a sequence of texts as a matrix."""
        return cast(Matrix, self._model.encode(texts))  # type: ignore

    def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> Matrix:
        """Split `texts` into batches of `batch_size` and encode each separately.

        Args:
            texts: Strings to encode.
            batch_size: Number of items per batch.
            progress: If True, use `tqdm` to show a progress bar for the process.

        Returns:
            The encoded vectors stacked as a single matrix.
        """
        batches = itertools.batched(texts, n=batch_size)

        if progress:
            batch_num = math.ceil(len(texts) / batch_size)
            batches = tqdm(batches, total=batch_num, desc="Batch text encoding")

        return np.vstack([self.encode_multi(batch) for batch in batches])


def similarities(vector: Vector, matrix: Matrix) -> npt.NDArray[np.float32]:
    """Calculate cosine similarity between a `vector` and a `matrix`.

    The `vector` will have shape `(dim,)`, and matrices `(N, dim)`, where N is the number
    of entries. The output should be a vector of floats with N elements.

    Raises:
        ValueError: if vector or matrix have the wrong shape.
    """
    # Preconditions
    if vector.ndim != 1:
        raise ValueError("vector must be 1D")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if vector.shape[0] != matrix.shape[1]:
        raise ValueError("vector and matrix dimensions must be compatible")

    dot_product = np.dot(matrix, vector)

    # Compute the L2 norms (magnitudes)
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    denominator = (vector_norm * matrix_norms) + epsilon

    similarities = (dot_product / denominator).astype(np.float32)

    # Postconditions
    assert similarities.ndim == 1, "similarities should be 1D"
    assert similarities.shape[0] == matrix.shape[0], (
        "similarities must have as many elements as matrix rows"
    )

    return similarities


def top_k_indices(vector: Vector, k: int) -> list[int]:
    """Get the indices of the top K elements in the vector, sorted descending."""
    return [int(x) for x in np.argsort(vector)[::-1][:k]]


class MatrixData(BaseModel):
    """Data object used to serialise a numpy matrix as JSON."""

    model_config = ConfigDict(frozen=True)

    shape: list[int]
    """Shape of the matrix as expected by numpy."""
    dtype: str
    """Underlying numpy data type of the matrix."""
    data: str
    """Matrix data encoded as base 64 bytes in UTF-8."""

    @classmethod
    def from_matrix(cls, matrix: Matrix) -> Self:
        """Serialise an embedding matrix to a base64-encoded byte string with metadata."""
        return cls(
            shape=list(matrix.shape),
            dtype=str(matrix.dtype),
            data=base64.b64encode(matrix.tobytes()).decode("utf-8"),
        )

    def to_matrix(self) -> Matrix:
        """Convert the serialised data back to a matrix."""
        bytes_data = base64.b64decode(self.data.encode("utf-8"))
        matrix = np.frombuffer(bytes_data, dtype=np.dtype(self.dtype))
        return matrix.reshape(self.shape)


def _get_best_device() -> str | None:
    """Use "CUDA" if available, then "MPS" (Apple Silicon), or whatever is the default."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None
