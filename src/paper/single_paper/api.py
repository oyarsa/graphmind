"""Public library API for evaluating paper novelty from arXiv titles.

This is the recommended entry point for using this package as a library (as
opposed to the `paper single`/`paper single-batch` CLI commands, which are
thin wrappers around these same functions).

Requires `SEMANTIC_SCHOLAR_API_KEY` and `OPENAI_API_KEY`/`GEMINI_API_KEY`
environment variables (or a `.env` file in the working directory).
"""

from __future__ import annotations

import asyncio
import dataclasses as dc
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from paper.gpt.openai_encoder import OpenAIEncoder
from paper.single_paper.pipeline import QueryType, process_paper_from_query
from paper.single_paper.s2_store import CachingS2Store, LiveS2Store, S2Store
from paper.util import ensure_envvar, progress
from paper.util.rate_limiter import get_limiter

if TYPE_CHECKING:
    from paper.gpt.extract_graph import EvaluationResult


@dc.dataclass(frozen=True)
class PaperEvaluationConfig:
    """Tunable pipeline parameters. Defaults match the CLI's defaults."""

    top_k_refs: int = 20
    num_recommendations: int = 30
    num_related: int = 2
    llm_model: str = "gpt-4o-mini"
    encoder_model: str = "text-embedding-3-small"
    seed: int = 0
    eval_prompt_key: str = "full-graph-structured"
    graph_prompt_key: str = "full"
    demonstrations_key: str = "orc_4"
    demo_prompt_key: str = "abstract"


@dc.dataclass(frozen=True)
class EvaluationError:
    """A paper that failed to evaluate, with why.

    A distinct type (not a bare `str`) so it's discriminable via `isinstance`
    when reading `PaperEvaluation.result`.
    """

    title: str
    message: str


@dc.dataclass(frozen=True)
class PaperEvaluation:
    """Outcome for one title: a completed result, or an error, never both."""

    title: str
    result: EvaluationResult | EvaluationError


_DEFAULT_CONFIG = PaperEvaluationConfig()


async def evaluate_paper(
    title: str,
    config: PaperEvaluationConfig = _DEFAULT_CONFIG,
    *,
    store: S2Store | None = None,
) -> EvaluationResult:
    """Evaluate a single paper's novelty from its arXiv title.

    Args:
        title: Paper title to search for on arXiv/Semantic Scholar.
        config: Tunable pipeline parameters.
        store: Semantic Scholar data access. Defaults to a live, uncached
            lookup; pass a `CachingS2Store` to reuse results across calls.

    Returns:
        Novelty evaluation for the paper.

    Raises:
        ValueError: If the paper isn't found on Semantic Scholar or arXiv, or
            no recommended papers or valid references are found.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.
    """
    limiter = get_limiter(1, 1)
    encoder = OpenAIEncoder(model=config.encoder_model)
    return await process_paper_from_query(
        title,
        QueryType.TITLE,
        config.top_k_refs,
        config.num_recommendations,
        config.num_related,
        config.llm_model,
        encoder,
        config.seed,
        limiter,
        config.eval_prompt_key,
        config.graph_prompt_key,
        config.demonstrations_key,
        config.demo_prompt_key,
        store=store,
    )


async def evaluate_papers(
    titles: Sequence[str],
    config: PaperEvaluationConfig = _DEFAULT_CONFIG,
    *,
    concurrency: int = 4,
    cache_path: Path | None = None,
) -> list[PaperEvaluation]:
    """Evaluate many papers concurrently, sharing one Semantic Scholar cache.

    Never raises for individual paper failures — check each result's
    `.result` type (`EvaluationResult` vs `EvaluationError`) instead, so one
    bad title doesn't lose the rest of the batch.

    Args:
        titles: Paper titles to search for and evaluate.
        config: Tunable pipeline parameters, shared by every paper.
        concurrency: Maximum papers processed concurrently.
        cache_path: If given, the Semantic Scholar cache persists to this
            file and is reused across calls/runs. If None, the cache only
            lives for the duration of this call.

    Returns:
        One `PaperEvaluation` per title, in the same order as `titles`.
    """
    s2_api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    limiter = get_limiter(1, 1)
    encoder = OpenAIEncoder(model=config.encoder_model)
    store = CachingS2Store(LiveS2Store(s2_api_key, limiter), cache_path)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(title: str) -> PaperEvaluation:
        async with semaphore:
            try:
                result = await process_paper_from_query(
                    title,
                    QueryType.TITLE,
                    config.top_k_refs,
                    config.num_recommendations,
                    config.num_related,
                    config.llm_model,
                    encoder,
                    config.seed,
                    limiter,
                    config.eval_prompt_key,
                    config.graph_prompt_key,
                    config.demonstrations_key,
                    config.demo_prompt_key,
                    store=store,
                )
            except (ValueError, RuntimeError) as e:
                return PaperEvaluation(
                    title=title, result=EvaluationError(title=title, message=str(e))
                )
        return PaperEvaluation(title=title, result=result)

    try:
        evaluations = await progress.gather(
            (run_one(title) for title in titles), desc="Evaluating papers"
        )
    finally:
        store.flush()

    return list(evaluations)
