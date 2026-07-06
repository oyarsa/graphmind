"""Tests for the pluggable embedding provider (OpenAI vs local)."""

from collections.abc import Sequence

import numpy as np
import pytest

from paper import embedding as emb
from paper.embedding import LocalEncoder, TextEncoder
from paper.gpt.openai_encoder import OpenAIEncoder
from paper.single_paper.api import EmbeddingProvider, build_encoder
from paper.types import Matrix, Vector


class _FakeSyncEncoder:
    """Stand-in for the heavy SentenceTransformers `Encoder`, recording calls."""

    def __init__(self, model_name: str = "fake", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[tuple[str, object]] = []

    def encode(self, text: str) -> Vector:
        self.calls.append(("encode", text))
        return np.zeros(4, dtype=np.float32)

    def encode_multi(self, texts: Sequence[str]) -> Matrix:
        self.calls.append(("encode_multi", list(texts)))
        return np.zeros((len(texts), 4), dtype=np.float32)

    def batch_encode(
        self, texts: Sequence[str], batch_size: int = 128, *, progress: bool = False
    ) -> Matrix:
        self.calls.append(("batch_encode", list(texts)))
        return np.zeros((len(texts), 4), dtype=np.float32)


def test_build_encoder_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI provider builds an OpenAIEncoder."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    encoder = build_encoder(EmbeddingProvider.OPENAI, None)
    assert isinstance(encoder, OpenAIEncoder)


def test_build_encoder_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local provider builds a LocalEncoder without loading a real model."""
    monkeypatch.setattr(emb, "Encoder", _FakeSyncEncoder)
    encoder = build_encoder(EmbeddingProvider.LOCAL, None)
    assert isinstance(encoder, LocalEncoder)


def test_build_encoder_accepts_plain_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_encoder` coerces plain provider strings to the enum."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(emb, "Encoder", _FakeSyncEncoder)
    assert isinstance(build_encoder("openai", None), OpenAIEncoder)  # type: ignore[arg-type]
    assert isinstance(build_encoder("local", None), LocalEncoder)  # type: ignore[arg-type]


def test_local_encoder_uses_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default model name is passed through to the underlying encoder."""
    monkeypatch.setattr(emb, "Encoder", _FakeSyncEncoder)
    encoder = build_encoder(EmbeddingProvider.LOCAL, "some-other-model")
    assert isinstance(encoder, LocalEncoder)
    inner = encoder._encoder  # type: ignore[attr-defined]
    assert isinstance(inner, _FakeSyncEncoder)
    assert inner.model_name == "some-other-model"


@pytest.mark.asyncio
async def test_local_encoder_bridges_to_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LocalEncoder delegates its async methods to the sync encoder's methods."""
    monkeypatch.setattr(emb, "Encoder", _FakeSyncEncoder)
    encoder = LocalEncoder("fake")
    inner: _FakeSyncEncoder = encoder._encoder  # type: ignore[assignment]

    vec = await encoder.encode("hello")
    mat = await encoder.encode_multi(["a", "b"])
    batched = await encoder.batch_encode(["a", "b", "c"], batch_size=2)

    assert vec.shape == (4,)
    assert mat.shape == (2, 4)
    assert batched.shape == (3, 4)
    assert [name for name, _ in inner.calls] == [
        "encode",
        "encode_multi",
        "batch_encode",
    ]


def test_encoders_satisfy_interface() -> None:
    """Both concrete encoders are recognised as TextEncoder implementations."""
    assert issubclass(OpenAIEncoder, TextEncoder)
    assert issubclass(LocalEncoder, TextEncoder)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_local_encoder_real_model() -> None:
    """The real local model produces sensibly-shaped float32 embeddings.

    Exercises the actual SentenceTransformers path end to end (needs the model
    available locally); gated behind --runslow since it loads a torch model.
    """
    encoder = LocalEncoder("all-MiniLM-L6-v2")

    vec = await encoder.encode("attention is all you need")
    mat = await encoder.encode_multi(["transformers", "recurrent networks"])
    batched = await encoder.batch_encode(["a", "b", "c"], batch_size=2)

    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert mat.shape == (2, vec.shape[0])
    assert batched.shape == (3, vec.shape[0])
