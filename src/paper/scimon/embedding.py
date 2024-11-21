"""Tools to generate embeddings from text using SentenceTransformers."""

import base64
import os
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

type Vector = npt.NDArray[np.float32]
type Matrix = npt.NDArray[np.float32]
type SentenceTransformerPool = dict[Literal["input", "output", "processes"], Any]


class Encoder:
    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        # `sentence_transformers` has a bug where they don't clean up their semaphores
        # properly, so we suppress this.
        _hard_suppress_warning("multiprocessing.resource_tracker", "UserWarning")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

        # Used for parallel processing.
        self._pool = self._model.start_multi_process_pool()

    def encode(self, text: str) -> Vector:
        """Encode text as a vector."""
        return cast(Vector, self._model.encode(text))  # type: ignore

    def encode_multi(self, texts: Sequence[str]) -> Matrix:
        """Clean and parallel encode multiple texts as vectors."""
        embeddings = self._model.encode_multi_process(texts, self._pool)  # type: ignore
        return cast(Matrix, embeddings)

    def __enter__(self) -> Self:
        """Start multiprocessing pool."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close multiprocessing pool."""
        self._model.stop_multi_process_pool(self._pool)


def _hard_suppress_warning(name: str, type: str) -> None:
    """Suppress warning by adding it to `PYTHONWARNINGS`."""
    old_warnings = os.environ.get("PYTHONWARNINGS")
    new_filter = f"ignore::{type}:{name}"
    os.environ["PYTHONWARNINGS"] = (
        f"{old_warnings},{new_filter}" if old_warnings else new_filter
    )


def similarities(vector: Vector, matrix: Matrix) -> npt.NDArray[np.float32]:
    """Calculate cosine similarity between a `vector` and a `matrix`.

    The `vector` will have shape (dim,), and matrices (N, dim), where N is the number of
    entries. The output should be a vector of floats with N elements.

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
    assert (
        similarities.shape[0] == matrix.shape[0]
    ), "similarities must have as many elements as matrix rows"

    return similarities


class MatrixData(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: Sequence[int]
    dtype: str
    data: str

    @classmethod
    def from_matrix(cls, matrix: Matrix) -> Self:
        """Serialise a embedding matrix to a base64-encoded byte string with metadata."""
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
