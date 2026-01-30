"""Graph extraction and novelty evaluation.

This module handles extracting graph representations from papers and evaluating
their novelty using GPT-based analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Self

if TYPE_CHECKING:
    from paper.semantic_scholar.model import S2Reference

from pydantic import Field

from paper import gpt
from paper.backend.model import BEST_OF_N
from paper.gpt.evaluate_paper import (
    EvidenceItem,
    GPTStructured,
    GPTStructuredRaw,
    fix_evaluated_rating,
)
from paper.gpt.evaluate_paper_graph import (
    PRIMARY_AREAS,
    format_eval_template,
    get_demonstrations,
)
from paper.gpt.graph_types.excerpts import GPTExcerpt
from paper.gpt.novelty_utils import get_novelty_probability
from paper.gpt.prompts.evaluate_graph import GRAPH_EVAL_USER_PROMPTS
from paper.gpt.prompts.extract_graph import GRAPH_EXTRACT_USER_PROMPTS
from paper.gpt.run_gpt import GPTResult, LLMClient
from paper.types import Immutable, PaperSource
from paper.util import atimer
from paper.util.serde import replace_fields

logger = logging.getLogger(__name__)

type ProgressCallback = Callable[[str], Awaitable[None]]

DEFAULT_UNKNOWN_AUTHORS = "Unknown authors"


def format_author_names(
    names: Sequence[str],
    *,
    max_display: int = 2,
    default: str = DEFAULT_UNKNOWN_AUTHORS,
) -> str:
    """Format a list of author names for display.

    Args:
        names: List of author name strings.
        max_display: Maximum number of authors to show before using "et al."
        default: String to return if no names are provided.

    Returns:
        Formatted author string like "Alice, Bob" or "Alice et al."
    """
    if not names:
        return default

    if len(names) <= max_display:
        return ", ".join(names)

    return f"{names[0]} et al."


def format_bibliography(references: Sequence[S2Reference]) -> str:
    """Format paper references as a bibliography section for GPT context.

    Creates a mapping from citation keys to paper metadata so GPT can naturally
    expand LaTeX citations to human-readable format (e.g., "Smith et al., 2023").

    Args:
        references: S2 references with citation keys.

    Returns:
        Formatted bibliography text, or empty string if no keys available.
    """
    lines: list[str] = []
    for ref in references:
        if not ref.citation_key:
            continue

        author_names = [a.name for a in ref.authors if a.name] if ref.authors else []
        authors_str = format_author_names(author_names)

        year_str = str(ref.year) if ref.year else "n.d."
        lines.append(f"[{ref.citation_key}] {ref.title}. {authors_str}, {year_str}.")

    return "\n".join(lines)


def format_graph_template_with_bibliography(
    prompt: gpt.PromptTemplate, paper: gpt.PeerReadAnnotated
) -> str:
    """Format graph extraction template with bibliography context.

    Extends the main text with a bibliography section so GPT can resolve
    citation keys to actual paper names and authors.

    Args:
        prompt: Graph extraction prompt template.
        paper: Annotated paper data.

    Returns:
        Formatted prompt string with bibliography included.
    """
    main_text = paper.paper.main_text

    # Add bibliography if references have citation keys
    if bibliography := format_bibliography(paper.paper.references):
        main_text = f"{main_text}\n\nBibliography:\n{bibliography}"

    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=main_text,
        primary_areas=", ".join(PRIMARY_AREAS),
    )


class EvaluationResult(Immutable):
    """Evaluation result with cost."""

    result: Annotated[gpt.GraphResult, Field(description="Evaluated graph result.")]
    cost: Annotated[float, Field(description="Total cost of using the LLM API.")]

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

    Returns:
        Extracted graph wrapped in GPTResult.
    """
    result = await client.run(
        GPTExcerpt,
        graph_prompt.system,
        format_graph_template_with_bibliography(graph_prompt, paper),
    )
    graph = result.map(
        lambda r: r.to_graph(paper.title, paper.abstract) if r else gpt.Graph.empty()
    )
    if graph.result.is_empty():
        logger.warning(f"Paper '{paper.title}': invalid Graph")

    return graph


def calculate_evidence_distribution(
    total_semantic: int,
    total_citations: int,
) -> tuple[int, int]:
    """Calculate how many semantic and citation evidence items to include.

    Priority:
    1. Take up to 3 semantic papers
    2. Take up to 2 citation papers
    3. If total < 5, fill with whichever source has more remaining

    Args:
        total_semantic: Total number of semantic evidence items available.
        total_citations: Total number of citation evidence items available.

    Returns:
        Tuple of (semantic_count, citation_count) to include.
    """
    # Start with preferred distribution: up to 3 semantic, up to 2 citations
    sem_count = min(total_semantic, 3)
    cit_count = min(total_citations, 2)

    # Fill remaining slots (up to 5 total) with whichever source has more left
    remaining_needed = 5 - sem_count - cit_count
    if remaining_needed > 0:
        remaining_sem = total_semantic - sem_count
        remaining_cit = total_citations - cit_count
        if remaining_sem > remaining_cit:
            sem_count += min(remaining_sem, remaining_needed)
        else:
            cit_count += min(remaining_cit, remaining_needed)

    return sem_count, cit_count


def _distribute_evidence(
    evidence: Sequence[EvidenceItem],
) -> tuple[EvidenceItem, ...]:
    """Distribute evidence: prefer 3 semantic + 2 citations, fill to max 5.

    Priority:
    1. Take up to 3 semantic papers
    2. Take up to 2 citation papers
    3. If total < 5, fill with whichever source has more remaining
    """
    semantic = [e for e in evidence if e.source == PaperSource.SEMANTIC]
    citations = [e for e in evidence if e.source == PaperSource.CITATIONS]

    sem_count, cit_count = calculate_evidence_distribution(
        len(semantic), len(citations)
    )

    return tuple(semantic[:sem_count] + citations[:cit_count])


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
        GPTStructuredRaw,
        eval_prompt.system,
        format_eval_template(eval_prompt, paper, graph, demonstrations),
    )
    eval = result.fix(GPTStructuredRaw.error)
    if not eval.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid evaluation result")

    prob = await get_novelty_probability(client, eval, best_of_n=BEST_OF_N)
    distributed = eval.map(
        lambda e: replace_fields(
            e,
            supporting_evidence=_distribute_evidence(e.supporting_evidence),
            contradictory_evidence=_distribute_evidence(e.contradictory_evidence),
        )
    )
    return distributed.lift(prob, lambda e, p: e.with_prob(p))


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
        y_pred=fix_evaluated_rating(evaluation).label,
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
