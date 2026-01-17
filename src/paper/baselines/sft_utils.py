"""Shared utilities for SFT baselines."""

from __future__ import annotations

import platform
import warnings
from typing import Annotated

import torch
from pydantic import Field

from paper.types import Immutable


class LoraConfig(Immutable):
    """Configuration for LoRA fine-tuning."""

    r: Annotated[int, Field(description="LoRA attention dimension")]
    alpha: Annotated[int, Field(description="LoRA alpha parameter")]
    dropout: Annotated[float, Field(description="Dropout probability for LoRA layers")]
    target_modules: Annotated[list[str], Field(description="Target modules for LoRA")]


def cuda_available() -> bool:
    """Check if we're on Linux and CUDA is available."""
    if platform.system() == "Darwin":
        return False
    return torch.cuda.is_available()


def suppress_hf_warnings() -> None:
    """Suppress some annoying and unnecessary HuggingFace warnings."""
    warnings.filterwarnings(
        "ignore", message=".*use_reentrant parameter should be passed explicitly.*"
    )
