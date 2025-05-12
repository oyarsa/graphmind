"""Comparison generation and loading for tournament evaluation.

This module handles generating new comparisons between items using an LLM,
as well as loading existing comparisons from previous runs.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from paper.gpt.evaluate_tournament.tournament import (
    ComparisonResult,
    EvaluationInput,
    InputFileType,
    MatchResult,
    MatchWinner,
    find_common_papers,
)
from paper.gpt.model import Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate
from paper.gpt.run_gpt import GPTResult, LLMClient, gpr_map, gpr_traverse
from paper.util import progress, sample
from paper.util.serde import load_data, load_data_single

logger = logging.getLogger(__name__)

# Constants
REQUEST_BATCH_SIZE = 100


def format_evaluation_prompt(
    metric: str,
    paper: EvaluationInput,
    rationale_a: str,
    rationale_b: str,
    prompt: PromptTemplate,
    metric_definitions: Mapping[str, str],
) -> str:
    """Format user prompt from paper data.

    Args:
        metric: The metric to focus on in the comparison.
        paper: Paper data (title, abstract, etc.).
        rationale_a: First rationale to compare.
        rationale_b: Second rationale to compare.
        prompt: Prompt template for the comparison.
        metric_definitions: Definitions for each metric.

    Returns:
        Comparison result wrapped in a GPTResult.
    """
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        rationale_a=rationale_a,
        rationale_b=rationale_b,
        metric=metric,
        definition=metric_definitions[metric],
    )


async def _compare_rationales(
    client: LLMClient,
    paper: EvaluationInput,
    rationale_a: str,
    rationale_b: str,
    metric: str,
    prompt: PromptTemplate,
    metric_definitions: Mapping[str, str],
) -> GPTResult[PromptResult[MatchResult]]:
    """Compare two rationales for the same paper using LLM.

    Args:
        client: LLM client.
        paper: Paper data (title, abstract, etc.).
        rationale_a: First rationale to compare.
        rationale_b: Second rationale to compare.
        metric: The metric to focus on in the comparison.
        prompt: Prompt template for the comparison.
        metric_definitions: Definitions for each metric.

    Returns:
        Comparison result and prompt wrapped in a GPTResult.
    """
    user_prompt_text = format_evaluation_prompt(
        metric, paper, rationale_a, rationale_b, prompt, metric_definitions
    )

    result = await client.run(MatchResult, prompt.system, user_prompt_text)
    return result.map(
        lambda r: PromptResult(
            item=r
            if r is not None
            # Default to TIE when LLM returns an error.
            else MatchResult(
                winner=MatchWinner.TIE,
                explanation="Comparison error. Defaulting to tie.",
            ),
            prompt=Prompt(system=prompt.system, user=user_prompt_text),
        )
    )


async def _run_all_comparisons(
    client: LLMClient,
    common_papers: Mapping[str, list[EvaluationInput]],
    metrics: Collection[str],
    item_names: Sequence[str],
    item_indices_pairs: Collection[tuple[int, int]],
    paper_ids: Collection[str],
    prompt: PromptTemplate,
    metric_definitions: Mapping[str, str],
) -> GPTResult[Sequence[PromptResult[ComparisonResult]]]:
    """Run all pairwise comparisons between items.

    Args:
        client: LLM client.
        common_papers: Papers from each item, grouped by paper ID.
        metrics: Metrics to evaluate.
        item_names: Names of the items being compared.
        item_indices_pairs: Pairs of item indices to compare.
        paper_ids: IDs of papers to use in comparisons.
        prompt: Prompt template for the comparison.
        metric_definitions: Definitions for each metric.

    Returns:
        List of comparison results.
    """
    total_comparisons = len(paper_ids) * len(item_indices_pairs) * len(metrics)
    logger.info(
        "Comparisons: Papers=%d * ItemPairs=%d * Metrics=%d = %d",
        len(paper_ids),
        len(item_indices_pairs),
        len(metrics),
        total_comparisons,
    )

    @dataclass(frozen=True, kw_only=True)
    class ComparisonSpec:
        """Specification for comparison task."""

        item_a: str
        item_b: str
        paper_id: str
        metric: str
        paper: EvaluationInput
        rationale_a: str
        rationale_b: str
        prompt: PromptTemplate

    comparison_specs: list[ComparisonSpec] = []

    for paper_id in paper_ids:
        papers = common_papers[paper_id]
        paper = papers[0]

        for metric in metrics:
            for i, j in item_indices_pairs:
                comparison_specs.append(
                    ComparisonSpec(
                        item_a=item_names[i],
                        item_b=item_names[j],
                        paper_id=paper_id,
                        metric=metric,
                        paper=paper,
                        rationale_a=papers[i].rationale,
                        rationale_b=papers[j].rationale,
                        prompt=prompt,
                    )
                )

    comparison_results: list[
        GPTResult[PromptResult[tuple[ComparisonSpec, MatchResult]]]
    ] = []

    with tqdm(
        total=len(comparison_specs),
        desc="Running pairwise comparisons",
        position=0,
        leave=True,
    ) as pbar_cmp:
        for batch_specs in itertools.batched(comparison_specs, REQUEST_BATCH_SIZE):
            tasks = [
                _compare_rationales(
                    client,
                    spec.paper,
                    spec.rationale_a,
                    spec.rationale_b,
                    spec.metric,
                    spec.prompt,
                    metric_definitions,
                )
                for spec in batch_specs
            ]
            batch_results = await progress.gather(
                tasks,
                desc="Running pairwise comparisons batch",
                position=1,
                leave=False,
            )
            comparison_results.extend(
                gpr_map(result, lambda r, spec=spec: (spec, r))
                for spec, result in zip(batch_specs, batch_results)
            )
            pbar_cmp.update(len(batch_specs))

    def transform(
        item: tuple[ComparisonSpec, MatchResult],
    ) -> ComparisonResult:
        spec, cmp = item
        return ComparisonResult(
            item_a=spec.item_a,
            item_b=spec.item_b,
            metric=spec.metric,
            paper=spec.paper,
            rationale_a=spec.rationale_a,
            rationale_b=spec.rationale_b,
            result=cmp,
        )

    return gpr_traverse(comparison_results, transform)


class RawComparisonOutput(BaseModel):
    """Raw comparisons output for serialization."""

    model_config = ConfigDict(frozen=True)

    item_names: Sequence[str]
    metrics: Sequence[str]
    seed: int
    comparisons: Sequence[PromptResult[ComparisonResult]]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class CachedResult[T]:
    """Result of a GPT request and its full API cost."""

    result: T

    @property
    def cost(self) -> float:
        """The cost of a cached result is always -1 to signal it didn't cost anything."""
        return -1


async def load_reused_comparisons(path: Path) -> CachedResult[RawComparisonOutput]:
    """Load comparison data from a previous run.

    Args:
        path: Path to the saved comparison data file.

    Returns:
        Cached result with loaded comparison data.
    """
    logger.info(f"Reusing comparison data from {path}")
    data = load_data_single(path, RawComparisonOutput)

    logger.info(
        f"Loaded {len(data.comparisons)} comparisons for {len(data.item_names)} models"
    )
    return CachedResult(result=data)


def _all_pairings[T](xs: Iterable[T]) -> list[tuple[T, T]]:
    """Create possible pairings of elements (A-B, A-C, B-C, B-A etc.). Order-sensitive."""
    return list(itertools.permutations(xs, 2))


def _load_evaluation_input(
    file_path: Path, file_type: InputFileType
) -> Sequence[EvaluationInput]:
    """Load evaluation input data from file.

    Args:
        file_path: Path to the input file.
        file_type: Type of the input file.

    Returns:
        Sequence of evaluation inputs.
    """
    match file_type:
        case InputFileType.GRAPH:
            from paper.gpt.extract_graph import GraphResult

            data = PromptResult.unwrap(load_data(file_path, PromptResult[GraphResult]))
        case InputFileType.PAPER:
            from paper.gpt.evaluate_paper import PaperResult

            data = PromptResult.unwrap(load_data(file_path, PromptResult[PaperResult]))
        case InputFileType.SUMM:
            from paper.gpt.model import PaperWithRelatedSummary

            data = PromptResult.unwrap(
                load_data(file_path, PromptResult[PaperWithRelatedSummary])
            )
        case InputFileType.RAW:
            from paper import peerread as pr

            data = load_data(file_path, pr.Paper)

    return data


async def generate_new_comparisons(
    client: LLMClient,
    inputs: Collection[tuple[Path, InputFileType]],
    model_names: Sequence[str],
    metrics: Sequence[str],
    metric_definitions: Mapping[str, str],
    limit: int,
    model: str,
    tournament_prompt_key: str,
    seed: int,
    algorithm: str,
    prompt: PromptTemplate,
) -> GPTResult[RawComparisonOutput]:
    """Generate new comparisons by running the LLM.

    Args:
        client: LLM client used to perform comparisons.
        inputs: List of (file_path, file_type) tuples.
        model_names: Names of the models.
        metrics: Metrics to evaluate.
        metric_definitions: Definitions for each metric.
        limit: Maximum number of papers to use.
        model: GPT model to use.
        tournament_prompt_key: Key for the comparison prompt.
        seed: Random seed.
        algorithm: Ranking algorithm to use.
        prompt: Prompt template to use for comparisons.

    Returns:
        The full result from the comparisons.

    Raises:
        ValueError if there are no common papers to compare.
    """
    paper_collections = [
        _load_evaluation_input(file_path, file_type) for file_path, file_type in inputs
    ]
    common_papers = find_common_papers(paper_collections)

    if not common_papers:
        raise ValueError(
            "No common papers found across all models. Tournament cannot proceed."
        )

    logger.info(
        f"Found {len(common_papers)} papers common to all {len(model_names)} models"
    )

    # Step 1: Run all pairwise comparisons
    paper_ids = sample(list(common_papers), limit)
    model_indices_pairs = _all_pairings(range(len(model_names)))

    comparisons_result = await _run_all_comparisons(
        client,
        common_papers,
        metrics,
        model_names,
        model_indices_pairs,
        paper_ids,
        prompt,
        metric_definitions,
    )

    return comparisons_result.map(
        lambda cmp: RawComparisonOutput(
            comparisons=cmp,
            item_names=model_names,
            metrics=metrics,
            seed=seed,
            metadata={
                "model": model,
                "prompt": tournament_prompt_key,
                "paper_count": len(paper_ids),
                "algorithm": algorithm,
            },
        )
    )
