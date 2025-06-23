"""Graph extraction and novelty evaluation.

This module handles extracting graph representations from papers and evaluating
their novelty using GPT-based analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Self

from paper import gpt
from paper.evaluation_metrics import TargetMode
from paper.gpt.evaluate_paper import GPTStructured, fix_evaluated_rating
from paper.gpt.evaluate_paper_graph import (
    GRAPH_EVAL_USER_PROMPTS,
    GRAPH_EXTRACT_USER_PROMPTS,
    format_eval_template,
    format_graph_template,
    get_demonstrations,
)
from paper.gpt.graph_types.full import GPTGraph
from paper.gpt.run_gpt import GPTResult, LLMClient
from paper.types import Immutable
from paper.util import atimer

logger = logging.getLogger(__name__)

type ProgressCallback = Callable[[str], Awaitable[None]]


class EvaluationResult(Immutable):
    """Evaluation result with cost."""

    result: gpt.GraphResult
    cost: float

    @classmethod
    def from_(cls, result: GPTResult[gpt.GraphResult]) -> Self:
        """Create EvaluationResult rom GPTResult+GraphResult."""
        return cls(result=result.result, cost=result.cost)


def get_prompts(
    eval_prompt_key: str, graph_prompt_key: str
) -> tuple[gpt.PromptTemplate, gpt.PromptTemplate]:
    """Retrieve evaluation and graph extraction prompts.

    Both must have system prompts. The eval prompt must have type GPTStructured.

    Args:
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.

    Returns:
        Tuple of (eval_prompt, graph_prompt).

    Raises:
        ValueError: If prompts are invalid.
    """
    eval_prompt = GRAPH_EVAL_USER_PROMPTS[eval_prompt_key]
    if not eval_prompt.system or eval_prompt.type_name != "GPTStructured":
        raise ValueError(f"Eval prompt {eval_prompt.name!r} is not valid.")

    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    if not graph_prompt.system:
        raise ValueError(f"Graph prompt {graph_prompt.name!r} is not valid.")

    return eval_prompt, graph_prompt


async def extract_graph_from_paper(
    paper: gpt.PeerReadAnnotated, client: LLMClient, graph_prompt: gpt.PromptTemplate
) -> GPTResult[gpt.Graph]:
    """Extract graph representation from a paper.

    Args:
        paper: Annotated paper data.
        client: LLM client for GPT API calls.
        graph_prompt: Graph extraction prompt template.
        title: Paper title.
        abstract: Paper abstract.

    Returns:
        Extracted graph wrapped in GPTResult.
    """
    result = await client.run(
        GPTGraph, graph_prompt.system, format_graph_template(graph_prompt, paper)
    )
    graph = result.map(
        lambda r: r.to_graph(paper.title, paper.abstract) if r else gpt.Graph.empty()
    )
    if graph.result.is_empty():
        logger.warning(f"Paper '{paper.title}': invalid Graph")

    return graph


async def evaluate_paper_graph_novelty(
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    client: LLMClient,
    eval_prompt: gpt.PromptTemplate,
    demonstrations: str,
) -> GPTResult[GPTStructured]:
    """Evaluate a paper's novelty using the extracted graph.

    Args:
        paper: Paper with related papers and summaries.
        graph: Extracted graph representation.
        client: LLM client for GPT API calls.
        eval_prompt: Evaluation prompt template.
        demonstrations: Demonstration examples.

    Returns:
        Evaluation result wrapped in GPTResult.
    """
    result = await client.run(
        GPTStructured,
        eval_prompt.system,
        format_eval_template(eval_prompt, paper, graph, demonstrations),
    )
    eval = result.map(lambda r: r or GPTStructured.error())
    if not eval.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid evaluation result")

    return eval


def construct_graph_result(
    paper: gpt.PaperWithRelatedSummary, graph: gpt.Graph, evaluation: GPTStructured
) -> gpt.GraphResult:
    """Construct the final graph result from components.

    Args:
        paper: Paper with related papers and summaries.
        graph: Extracted graph representation.
        evaluation: Novelty evaluation result.

    Returns:
        Complete GraphResult.
    """
    result = gpt.PaperResult.from_s2peer(
        paper=paper.paper.paper,
        y_pred=fix_evaluated_rating(evaluation, TargetMode.BIN).label,
        rationale_pred=evaluation.rationale,
        structured_evaluation=evaluation,
    )
    return gpt.GraphResult.from_annotated(annotated=paper, graph=graph, result=result)


async def evaluate_paper_with_graph(
    paper: gpt.PaperWithRelatedSummary,
    client: LLMClient,
    eval_prompt_key: str,
    graph_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
    *,
    callback: ProgressCallback | None = None,
) -> GPTResult[gpt.GraphResult]:
    """Evaluate a paper's novelty using graph extraction and related papers.

    Args:
        paper: Paper with related papers and summaries from PETER pipeline.
        client: LLM client for GPT API calls.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.
        callback: Optional callback function to call with phase names after completion.

    Returns:
        GraphResult with novelty evaluation wrapped in GPTResult.
    """
    eval_prompt, graph_prompt = get_prompts(eval_prompt_key, graph_prompt_key)
    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    if callback:
        await callback("Extracting graph representation")

    graph_result = await atimer(
        extract_graph_from_paper(paper.paper, client, graph_prompt), 3
    )

    if callback:
        await callback("Evaluating novelty")

    eval_result = await atimer(
        evaluate_paper_graph_novelty(
            paper, graph_result.result, client, eval_prompt, demonstrations
        ),
        3,
    )

    return GPTResult(
        result=construct_graph_result(paper, graph_result.result, eval_result.result),
        cost=graph_result.cost + eval_result.cost,
    )
