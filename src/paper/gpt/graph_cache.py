"""Content-addressed caching for graph extraction results.

This module provides caching functionality to avoid re-extracting graphs when running
experiments with the same configuration. The cache key is computed from the parameters
that affect graph extraction (model, temperature, graph_prompt_key, seed, input_file).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from paper.gpt.model import Graph
from paper.util.serde import Compress, load_data, read_file_bytes, save_data

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def compute_cache_key(
    model: str,
    temperature: float,
    graph_prompt_key: str,
    seed: int,
    input_file: Path,
) -> str:
    """Compute SHA256 hash of cache key parameters.

    The cache key uniquely identifies a graph extraction configuration. If any of these
    parameters change, a new cache key is generated and graphs are re-extracted.

    Args:
        model: LLM model used for graph extraction.
        temperature: Temperature for graph extraction (typically 0.0).
        graph_prompt_key: Key identifying the graph extraction prompt.
        seed: Random seed for reproducibility.
        input_file: Path to the input papers file.

    Returns:
        First 16 characters of the SHA256 hash.
    """
    file_hash = hashlib.sha256(read_file_bytes(input_file)).hexdigest()

    key_parts = [
        f"model={model}",
        f"temperature={temperature}",
        f"graph_prompt_key={graph_prompt_key}",
        f"seed={seed}",
        f"input_file_hash={file_hash}",
    ]
    key_string = "|".join(key_parts)

    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def load_cached_graphs(cache_path: Path) -> Mapping[str, Graph] | None:
    """Load graphs from cache, return None if not found.

    Args:
        cache_path: Path to cache directory (from get_cache_path()).

    Returns:
        Dictionary mapping paper IDs to Graph objects, or None if cache doesn't exist.
    """
    graphs_file = cache_path / "graphs.json.zst"

    if not graphs_file.exists():
        return None

    try:
        graphs = load_data(graphs_file, Graph)
    except Exception as e:
        logger.warning("Failed to load cache from %s: %s", cache_path, e)
        return None
    else:
        result = {g.id: g for g in graphs}
        logger.info("Loaded %d graphs from cache: %s", len(result), cache_path)
        return result


def save_graphs_to_cache(
    cache_path: Path,
    graphs: Mapping[str, Graph],
    params: dict[str, str],
) -> None:
    """Save graphs to cache with params for inspection.

    Args:
        cache_path: Path to cache directory (from get_cache_path()).
        graphs: Dictionary mapping paper IDs to Graph objects.
        params: Dictionary of cache key parameters for human inspection.
    """
    cache_path.mkdir(parents=True, exist_ok=True)

    graphs_list = list(graphs.values())
    save_data(cache_path / "graphs.json.zst", graphs_list, compress=Compress.ZSTD)
    save_data(cache_path / "params.json", params, compress=Compress.NONE)

    logger.info("Saved %d graphs to cache: %s", len(graphs_list), cache_path)
