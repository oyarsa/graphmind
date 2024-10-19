"""Infamous utility module for stuff I can't place anywhere else."""

import logging
import os
import time
from typing import Any, Self

import colorlog
import spacy
from thefuzz import fuzz  # type: ignore


def fuzzy_ratio(s1: str, s2: str) -> int:
    """Calculates the fuzzy matching ratio between s1 and s2 as integer in 0-100.

    Type-safe wrapper around thefuzz.fuzz.ratio.
    """
    return fuzz.ratio(s1, s2)  # type: ignore


class BlockTimer:
    """Track the time elapsed during its block."""

    def __init__(self) -> None:
        self._start_time = 0
        self._elapsed_seconds = 0

    def __enter__(self) -> Self:
        """Start the timer when entering the context."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Stop the timer when exiting the context."""
        self._elapsed_seconds = time.perf_counter() - self._start_time

    @property
    def seconds(self) -> float:
        """Return the elapsed time in seconds."""
        return self._elapsed_seconds

    @property
    def human(self) -> str:
        """Return the elapsed time in a human-readable format."""
        seconds = self._elapsed_seconds
        units = [("d", 86400), ("h", 3600), ("m", 60)]
        parts: list[str] = []

        for name, count in units:
            value, seconds = divmod(seconds, count)
            if value >= 1:
                parts.append(f"{int(value)}{name}")

        if seconds > 0 or not parts:
            parts.append(f"{seconds:.2f}s")

        return " ".join(parts)


def setup_logging(logger: logging.Logger) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = colorlog.StreamHandler()

    fmt = "%(log_color)s%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(colorlog.ColoredFormatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(handler)


def load_spacy_model(model_name: str) -> spacy.language.Language:
    """Ensure that a spaCy model is available, downloading it if necessary.

    NB: spaCy neeeds `pip` to download the model, so make sure your environment has it
    (e.g. explicitly add `pip` as a dependency if using uv or poetry).

    Args:
        model_name: The name of the spaCy model to load (e.g., "en_core_web_sm")
    """
    if not spacy.util.is_package(model_name):
        print(f"Model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)  # type: ignore

    return spacy.load(model_name)
