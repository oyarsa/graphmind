"""Graph extraction and novelty evaluation using multiple perspectives.

This module handles extracting graph representations from papers and evaluating
their novelty using GPT-based analysis.

Each paper is evaluated using multiple perspecitves (e.g. technical contributions,
problem setup, experiment execution) and then aggregated to get a final label.
"""

from __future__ import annotations

import asyncio
import dataclasses as dc
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import Field, computed_field

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import semantic_scholar as s2
from paper.evaluation_metrics import TargetMode
from paper.gpt.evaluate_paper_graph import (
    GRAPH_EXTRACT_USER_PROMPTS,
    format_graph_template,
    format_related,
    get_demonstrations,
)
from paper.gpt.graph_types.excerpts import GPTExcerpt
from paper.gpt.model import RATIONALE_ERROR
from paper.gpt.prompts import load_prompts
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_sequence, gpt_unit
from paper.single_paper.paper_retrieval import get_paper_from_arxiv_id
from paper.single_paper.pipeline import annotate_paper_pipeline
from paper.types import Immutable, PaperProxy, PaperSource
from paper.util import atimer, ensure_envvar
from paper.util.serde import replace_fields

if TYPE_CHECKING:
    from paper.util.rate_limiter import Limiter

logger = logging.getLogger(__name__)

type ProgressCallback = Callable[[str], Awaitable[None]]

GRAPH_EVAL_MULTI_USER_PROMPTS = load_prompts("evaluate_graph_multi")
GRAPH_EVAL_SUMM_USER_PROMPTS = load_prompts("summarise_perspectives")


class EvidenceItemMulti(Immutable):
    """Evidence item for multi-perspective evaluation."""

    text: Annotated[
        str, Field(description="Piece of evidence or comparison from related papers.")
    ]
    paper_id: Annotated[
        str | None,
        Field(description="ID of the paper from which evidence is extracted."),
    ]
    paper_title: Annotated[
        str | None,
        Field(description="Title of the paper from which evidence is extracted."),
    ]
    source: Annotated[
        PaperSource | None,
        Field(
            description="Source of the related paper (citations or semantic similarity)."
        ),
    ]


class GPTStructuredMulti(Immutable):
    """Structured evaluation of paper novelty for multi-perspective analysis."""

    paper_summary: Annotated[
        str,
        Field(
            description="Brief summary of the paper's main contributions and approach."
        ),
    ]
    supporting_evidence: Annotated[
        Sequence[EvidenceItemMulti],
        Field(
            description="List of evidence from related papers that support the paper's"
            " novelty."
        ),
    ]
    contradictory_evidence: Annotated[
        Sequence[EvidenceItemMulti],
        Field(
            description="List of evidence from related papers that contradict the"
            " paper's novelty."
        ),
    ]
    key_comparisons: Annotated[
        Sequence[str],
        Field(
            description="Key technical comparisons that influenced the novelty decision."
        ),
    ]
    conclusion: Annotated[
        str,
        Field(
            description="Final assessment of the paper's novelty based on the evidence."
        ),
    ]
    label: Annotated[
        int,
        Field(description="1 if the paper is novel, or 0 if it's not novel."),
    ]

    @computed_field
    @property
    def rationale(self) -> str:
        """Derive textual rationale from structured components."""
        sections = [
            f"Paper Summary: {self.paper_summary}",
            "",
            "Supporting Evidence:",
        ]

        for evidence in self.supporting_evidence:
            evidence_text = f"- {evidence.text}"
            if evidence.paper_title:
                evidence_text += f" (from: {evidence.paper_title})"
            sections.append(evidence_text)

        sections.extend(["", "Contradictory Evidence:"])

        for evidence in self.contradictory_evidence:
            evidence_text = f"- {evidence.text}"
            if evidence.paper_title:
                evidence_text += f" (from: {evidence.paper_title})"
            sections.append(evidence_text)

        if self.key_comparisons:
            sections.extend(["", "Key Comparisons:"])
            sections.extend([f"- {comp}" for comp in self.key_comparisons])

        sections.extend(["", f"Conclusion: {self.conclusion}"])
        return "\n".join(sections)

    @classmethod
    def error(cls) -> Self:
        """Output value for when there's an error."""
        return cls(
            paper_summary=RATIONALE_ERROR,
            supporting_evidence=[],
            contradictory_evidence=[],
            key_comparisons=[],
            conclusion=RATIONALE_ERROR,
            label=0,
        )

    def is_valid(self) -> bool:
        """Check if instance is valid."""
        return RATIONALE_ERROR not in {self.paper_summary, self.conclusion}

    def to_text(self) -> str:
        """Convert structured information to text format for inclusion in prompts."""
        return self.rationale


@dc.dataclass(frozen=True, kw_only=True)
class _ChosenPrompts:
    """Prompt templates from user-picked keys."""

    eval: gpt.PromptTemplate
    graph: gpt.PromptTemplate
    summ: gpt.PromptTemplate
    structured: gpt.PromptTemplate


def get_prompts(
    eval_prompt_key: str,
    graph_prompt_key: str,
    summ_prompt_key: str,
    structured_prompt_key: str,
) -> _ChosenPrompts:
    """Get prompts used for the whole multi-perspective evaluation pipeline.

    Evaluation, graph extraction, structured and perspective summarisation prompts.
    Both must have system prompts. The eval prompt must have type GPTEvalMulti.

    Args:
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        summ_prompt_key: Key for perspective summarisation prompt template.
        structured_prompt_key: Key for structured extraction prompt template.

    Returns:
        Chosen prompts.

    Raises:
        ValueError: If prompts are invalid.
    """
    eval_prompt = GRAPH_EVAL_MULTI_USER_PROMPTS[eval_prompt_key]
    if not eval_prompt.system or eval_prompt.type_name != "GPTEvalMulti":
        raise ValueError(f"Eval prompt {eval_prompt.name!r} is not valid.")

    graph_prompt = GRAPH_EXTRACT_USER_PROMPTS[graph_prompt_key]
    if not graph_prompt.system:
        raise ValueError(f"Graph prompt {graph_prompt.name!r} is not valid.")

    summ_prompt = GRAPH_EVAL_SUMM_USER_PROMPTS[summ_prompt_key]
    if not summ_prompt.system:
        raise ValueError(f"summ prompt {summ_prompt.name!r} is not valid.")

    structured_prompt = GRAPH_EVAL_MULTI_USER_PROMPTS[structured_prompt_key]
    if (
        not structured_prompt.system
        or structured_prompt.type_name != "GPTStructuredMulti"
    ):
        raise ValueError(f"Structured prompt {structured_prompt.name!r} is not valid.")

    return _ChosenPrompts(
        eval=eval_prompt,
        graph=graph_prompt,
        summ=summ_prompt,
        structured=structured_prompt,
    )


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
        GPTExcerpt, graph_prompt.system, format_graph_template(graph_prompt, paper)
    )
    graph = result.map(
        lambda r: r.to_graph(paper.title, paper.abstract) if r else gpt.Graph.empty()
    )
    if graph.result.is_empty():
        logger.warning(f"Paper '{paper.title}': invalid Graph")

    return graph


async def extract_structured_from_paper(
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    client: LLMClient,
    structured_prompt: gpt.PromptTemplate,
    demonstrations: str,
    sources: set[PaperSource] | None = None,
) -> GPTResult[GPTStructuredMulti]:
    """Extract structured information from a paper with graph and related papers.

    Args:
        paper: Paper with related papers and summaries.
        graph: Extracted graph representation.
        client: LLM client for GPT API calls.
        structured_prompt: Structured extraction prompt template.
        demonstrations: Demonstration examples.
        sources: Paper sources to include in evaluation.

    Returns:
        Structured evaluation wrapped in GPTResult.
    """
    if sources is None:
        sources = set(PaperSource)

    prompt_text = format_struct_prompt(
        structured_prompt, paper, graph, demonstrations, sources
    )

    result = await client.run(
        GPTStructuredMulti,
        structured_prompt.system,
        prompt_text,
    )

    structured = result.fix(GPTStructuredMulti.error)
    if not structured.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid structured extraction")

    return structured


def format_struct_prompt(
    prompt: gpt.PromptTemplate,
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    demonstrations: str,
    sources: set[PaperSource],
) -> str:
    """Format structured extraction template using the paper graph and related papers."""
    related = [p for p in paper.related if p.source in sources]

    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        demonstrations=demonstrations,
        positive=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(),
    )


async def evaluate_paper_with_graph_multi(
    paper: gpt.PaperWithRelatedSummary,
    client: LLMClient,
    eval_prompt_key: str,
    graph_prompt_key: str,
    summ_prompt_key: str,
    struct_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
    perspectives: Sequence[str] | None,
    sources: set[PaperSource] | None,
    *,
    callback: ProgressCallback | None = None,
) -> GPTResult[GraphResultMulti]:
    """Evaluate a paper's multi-perspective novelty using graph extraction and related.

    Args:
        paper: Paper with related papers and summaries from PETER pipeline.
        client: LLM client for GPT API calls.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        summ_prompt_key: Key for perspective summarisation prompt template.
        struct_prompt_key: Key for structured extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.
        perspectives: List of perspectives to evaluate. If None, use all perspectives.
        sources: Paper sources to include in evaluation. If None, use all sources.
        callback: Optional callback function to call with phase names after completion.

    Returns:
        GraphResultMulti with novelty evaluation wrapped in GPTResult.
    """
    if not perspectives:
        perspectives = list(EVAL_MULTI_PERSPECTIVES)
    _raise_if_invalid_perspectives(perspectives)

    if not sources:
        sources = set(PaperSource)

    prompts = get_prompts(
        eval_prompt_key, graph_prompt_key, summ_prompt_key, struct_prompt_key
    )
    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    if callback:
        await callback("Extracting graph representation")

    graph_result = await atimer(
        extract_graph_from_paper(paper.paper, client, prompts.graph), 3
    )

    if callback:
        await callback("Extracting structured information")

    structured_result = await atimer(
        extract_structured_from_paper(
            paper,
            graph_result.result,
            client,
            prompts.structured,
            demonstrations,
        ),
        3,
    )

    if callback:
        await callback("Evaluating novelty")

    eval_result = await atimer(
        evaluate_paper_graph_novelty_multi(
            paper,
            graph_result.result,
            structured_result.result,
            client,
            prompts.eval,
            prompts.summ,
            perspectives,
            sources,
            demonstrations,
        ),
        3,
    )

    return GPTResult(
        result=construct_graph_result_multi(
            paper, graph_result.result, eval_result.result
        ),
        cost=graph_result.cost + structured_result.cost + eval_result.cost,
    )


def _raise_if_invalid_perspectives(perspective_keys: Sequence[str]) -> None:
    """Raise ValueError if any perspective keys are invalid.

    Valid keys are from `EVAL_MULTI_PERSPECTIVES`.
    """
    unavailable = [p for p in perspective_keys if p not in EVAL_MULTI_PERSPECTIVES]
    if not unavailable:
        return

    raise ValueError(
        f"Invalid perspectives: {unavailable}. Available: {list(EVAL_MULTI_PERSPECTIVES)}"
    )


async def evaluate_paper_graph_novelty_multi(
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    structured: GPTStructuredMulti,
    client: LLMClient,
    eval_prompt: gpt.PromptTemplate,
    summ_prompt: gpt.PromptTemplate,
    perspective_keys: Sequence[str],
    sources: set[PaperSource],
    demonstrations: str,
) -> GPTResult[GPTEvalMultiResult]:
    """Evaluate a paper's multi-dimensional novelty using the extracted graph.

    Evaluate each perspective in `perspective_keys` separately, then aggregates the
    results in a single place.

    Args:
        paper: Paper with related papers and summaries.
        graph: Extracted graph representation.
        structured: Structured information extracted from the paper.
        client: LLM client for GPT API calls.
        eval_prompt: Evaluation prompt template.
        summ_prompt: Perspective summarisation prompt template.
        perspective_keys: List of perspective (keys) to evaluate. Must be available on
            `EVAL_MULTI_PERSPECTIVES`.
        sources: Paper sources to include in evaluation.
        demonstrations: Demonstration examples.

    Returns:
        Multi-dimensional evaluation result wrapped in GPTResult.
    """
    evals = await asyncio.gather(
        *(
            eval_perspective(
                paper,
                graph,
                structured,
                client,
                eval_prompt,
                demonstrations,
                sources,
                perspective=EVAL_MULTI_PERSPECTIVES[perspective_key],
            )
            for perspective_key in perspective_keys
        )
    )

    return await gpt_sequence(evals).abind(
        lambda ps: _summarise_perspectives(client, summ_prompt, structured, ps)
    )


async def _summarise_perspectives(
    client: LLMClient,
    prompt: gpt.PromptTemplate,
    structured: GPTStructuredMulti,
    perspectives: Sequence[GPTEvalPerspective],
) -> GPTResult[GPTEvalMultiResult]:
    """Summarise multiple perspectives into a single overall evaluation result.

    Uses only valid perspectives (see `GPTEvalPerspective.is_valid`). If none of the
    perspectives are valid, returns an error result (see `GPTEvalMultiResult.error`).

    Args:
        client: LLM client for GPT API calls.
        prompt: Perspective summarisation prompt template.
        perspectives: List of perspective evaluation results.
        structured: Structured information extracted from the paper.

    Returns:
        Multi-dimensional evaluation result wrapped in GPTResult.
    """
    perspectives_valid = [p for p in perspectives if p.is_valid()]
    if not perspectives_valid:
        return gpt_unit(GPTEvalMultiResult.error())

    summary = await summarise_perspectives(client, prompt, perspectives_valid)
    return summary.map(
        lambda s: GPTEvalMultiResult.from_summary(perspectives_valid, s, structured)
    )


class PerspectiveSummary(Immutable):
    """Summary of all perspectives for multi-perspective evaluation."""

    summary: str
    label: int

    @classmethod
    def error(cls) -> Self:
        """Return an error summary placeholder result."""
        return cls(summary=RATIONALE_ERROR, label=0)

    def is_valid(self) -> bool:
        """Check if the summary result is valid."""
        return self.summary != RATIONALE_ERROR


async def summarise_perspectives(
    client: LLMClient,
    prompt: gpt.PromptTemplate,
    perspective_results: Sequence[GPTEvalPerspective],
) -> GPTResult[PerspectiveSummary]:
    """Summarise all perspective rationales in a single text."""

    result = await client.run(
        PerspectiveSummary,
        prompt.system,
        format_perspective_summary_template(prompt, perspective_results),
    )
    eval = result.fix(PerspectiveSummary.error)

    if not eval.result.is_valid():
        logger.warning("Invalid perspective summarisation result.")

    return eval


def format_perspective_summary_template(
    prompt: gpt.PromptTemplate, perspective_results: Sequence[GPTEvalPerspective]
) -> str:
    """Format perspective summary template using the perspective results."""
    return prompt.template.format(
        perspectives="\n".join(p.to_text() for p in perspective_results)
    )


@dc.dataclass(frozen=True, kw_only=True)
class PerspectiveInfo:
    """Information about a perspective used for multi-perspective evaluation."""

    name: str
    description: str

    def to_text(self) -> str:
        """Format perspective as text for inclusion in prompts."""
        return f"{self.name}: {self.description}"


EVAL_MULTI_PERSPECTIVES: Mapping[str, PerspectiveInfo] = {
    "technical_contributions": PerspectiveInfo(
        name="Technical Contributions",
        description=(
            "Assess the novelty of the paper's technical contributions, such as new "
            "algorithms, models, or methods. Consider how these contributions differ "
            "from existing work and their potential impact on the field."
        ),
    ),
    "problem_setup": PerspectiveInfo(
        name="Problem Setup",
        description=(
            "Evaluate the novelty of the problem setup addressed by the paper. "
            "Consider whether the problem is new or if it presents a novel angle "
            "on an existing problem. Assess the significance and relevance of "
            "the problem within its field."
        ),
    ),
    "experiment_execution": PerspectiveInfo(
        name="Experiment Execution",
        description=(
            "Analyze the novelty of the experimental execution in the paper. "
            "Consider the design of experiments, methodologies used, and how "
            "these contribute to new insights or findings. Evaluate whether "
            "the experiments are innovative and if they effectively address "
            "the research questions posed."
        ),
    ),
}


async def eval_perspective(
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    structured: GPTStructuredMulti,
    client: LLMClient,
    prompt: gpt.PromptTemplate,
    demonstrations: str,
    sources: set[PaperSource],
    perspective: PerspectiveInfo,
) -> GPTResult[GPTEvalPerspective]:
    """Evaluate a paper's novelty from a single perspective."""
    result = await client.run(
        GPTEvalPerspective,
        prompt.system,
        format_eval_template_multi(
            prompt, paper, graph, structured, demonstrations, perspective, sources
        ),
    )
    eval = result.fix(GPTEvalPerspective.error)

    if not eval.result.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid evaluation result")

    return eval


def format_eval_template_multi(
    prompt: gpt.PromptTemplate,
    paper: gpt.PaperWithRelatedSummary,
    graph: gpt.Graph,
    structured: GPTStructuredMulti,
    demonstrations: str,
    perspective: PerspectiveInfo,
    sources: set[PaperSource],
) -> str:
    """Format evaluation template.

    Uses the paper graph, structured info and related papers.
    """

    related = [p for p in paper.related if p.source in sources]
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        demonstrations=demonstrations,
        perspective=perspective.to_text(),
        positive=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
        graph=graph.to_text(),
        structured=structured.to_text(),
        approval=paper.paper.paper.approval,
    )


class GPTEvalMultiResult(Immutable):
    """Evaluation result with multi-perspective evaluation."""

    rationale: str
    """Overall evaluation rationale from combined perspectives."""
    label: int
    """Final binary novelty label."""
    probability: float
    """Probability of being novel based on the multi perspectives."""
    perspectives: Sequence[GPTEvalPerspective]
    """Collection of perspectives used."""
    structured: GPTStructuredMulti
    """Structured information extracted from the paper."""

    @classmethod
    def error(cls) -> Self:
        """Return an error evaluation placeholder result."""
        return cls(
            rationale=RATIONALE_ERROR,
            label=0,
            probability=0.0,
            perspectives=[],
            structured=GPTStructuredMulti.error(),
        )

    @classmethod
    def from_summary(
        cls,
        perspectives: Sequence[GPTEvalPerspective],
        summary: PerspectiveSummary,
        structured: GPTStructuredMulti,
    ) -> Self:
        """Create GPTEvalMulti from multiple GPTEvalSingle perspectives."""
        probability = sum(p.label for p in perspectives) / len(perspectives)

        return cls(
            rationale=summary.summary,
            label=summary.label,
            probability=probability,
            perspectives=perspectives,
            structured=structured,
        )

    def fix_label(self, target_mode: TargetMode = TargetMode.BIN) -> Self:
        """Fix the label to be valid for the given target mode."""
        if self.label in target_mode.labels():
            return self

        logger.warning(
            "Invalid label: %d. Converting to %s", self.label, target_mode.labels()
        )
        return replace_fields(self, label=0)


class GPTEvalPerspective(Immutable):
    """Evaluation result from a given perspective. Raw version without the name.

    Use `with_name` to create a complete object with the evaluation result and the
    original perspective name.
    """

    rationale: Annotated[
        str, Field(description="Rationale for given label for this perspective.")
    ]
    label: Annotated[
        int, Field(description="Novelty label given for this perspective, 0 or 1.")
    ]
    name: Annotated[
        str, Field(description="Name of the perspective used for this evaluation.")
    ]

    @classmethod
    def error(cls) -> Self:
        """Return an error evaluation placeholder result."""
        return cls(rationale=RATIONALE_ERROR, label=0, name=RATIONALE_ERROR)

    def is_valid(self) -> bool:
        """Check if the evaluation result is valid."""
        return self.rationale != RATIONALE_ERROR

    def to_text(self) -> str:
        """Convert evaluation result to text for LLM prompt."""
        return f"{self.name} - Label: {self.label}\nRationale: {self.rationale}"


class PaperResultMulti(s2.PaperWithS2Refs):
    """PeerRead paper with added novelty evaluation ground truth and GPT prediction."""

    y_true: Annotated[int, Field(description="Human annotation")]
    rationale_true: Annotated[str, Field(description="Human rationale annotation")]
    evaluation: GPTEvalMultiResult

    @computed_field
    @property
    def y_pred(self) -> int:
        """Model prediction."""
        return self.evaluation.label

    @computed_field
    @property
    def rationale_pred(self) -> str:
        """Model predicted rationale."""
        return self.evaluation.rationale

    @classmethod
    def from_s2peer(
        cls, paper: s2.PaperWithS2Refs, evaluation: GPTEvalMultiResult
    ) -> Self:
        """Construct `PaperResult` from the original paper and model predictions."""
        return cls.model_validate(
            paper.model_dump()
            | {
                "y_true": paper.label,
                "rationale_true": paper.rationale,
                "evaluation": evaluation,
            }
        )


class GraphResultMulti(Immutable, PaperProxy[PaperResultMulti]):
    """Extracted graph and paper evaluation results."""

    graph: gpt.Graph
    paper: PaperResultMulti
    related: Sequence[gpt.PaperRelatedSummarised]

    terms: gpt.PaperTerms
    background: str
    target: str

    @property
    def rationale_pred(self) -> str:
        """Predicted rationale from the underlying paper result."""
        return self.paper.rationale_pred

    @classmethod
    def from_annotated(
        cls,
        annotated: gpt.PaperWithRelatedSummary,
        result: PaperResultMulti,
        graph: gpt.Graph,
    ) -> Self:
        """Create GraphResultMulti with annotation data."""
        return cls(
            graph=graph,
            paper=result,
            related=annotated.related,
            terms=annotated.terms,
            background=annotated.background,
            target=annotated.target,
        )


def construct_graph_result_multi(
    paper: gpt.PaperWithRelatedSummary, graph: gpt.Graph, evaluation: GPTEvalMultiResult
) -> GraphResultMulti:
    """Construct the final graph result from components for multi-perspective evaluation.

    Args:
        paper: Paper with related papers and summaries and multi-perspective evaluation.
        graph: Extracted graph representation.
        evaluation: Novelty evaluation result.

    Returns:
        Complete GraphResultMulti.
    """
    result = PaperResultMulti.from_s2peer(
        paper=paper.paper.paper,
        evaluation=evaluation.fix_label(TargetMode.BIN),
    )
    return GraphResultMulti.from_annotated(annotated=paper, graph=graph, result=result)


class EvaluationResultMulti(Immutable):
    """Evaluation result with cost."""

    result: Annotated[GraphResultMulti, Field(description="Evaluated graph result.")]
    cost: Annotated[float, Field(description="Total cost of using the LLM API.")]

    @classmethod
    def from_(cls, result: GPTResult[GraphResultMulti]) -> Self:
        """Create EvaluationResult rom GPTResult+GraphResult."""
        return cls(result=result.result, cost=result.cost)


async def process_paper_from_selection_multi(
    client: LLMClient,
    title: str,
    arxiv_id: str,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    limiter: Limiter,
    encoder: emb.Encoder,
    eval_prompt_key: str,
    graph_prompt_key: str,
    summ_prompt_key: str,
    struct_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
    *,
    sources: set[PaperSource] | None = None,
    perspectives: Sequence[str] | None = None,
    filter_by_date: bool = False,
    callback: ProgressCallback | None = None,
) -> EvaluationResultMulti:
    """Process a paper from pre-selected title and arXiv ID using multiple perspectives.

    This function is designed for API usage where the user has already selected a paper
    from search results and has both the title and arXiv ID available. It provides the
    same complete end-to-end processing as `process_paper_from_query` but skips the
    search/selection phase.

    The function:
    1. Retrieves paper from arXiv using the provided ID
    2. Fetches S2 metadata using the title
    3. Processes through the complete PETER pipeline including:
       - S2 reference enhancement with top-k semantic similarity filtering
       - GPT-based annotation extraction (key terms, background, target)
       - Citation context classification (positive/negative polarity)
       - Related paper discovery via citations and semantic matching
       - GPT-generated summaries of related papers
    4. Extracts a graph representation
    5. Derives structured info combining the graph representation and related papers.
    6. Uses graph and structured info to evaluate novelty using multiple perspectives,
        combining their results in a single output value.

    Args:
        client: LLMClient to use for annotation and evaluation.
        title: Paper title (used for S2 lookups and display).
        arxiv_id: arXiv ID of the paper (e.g. "2301.00234").
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        encoder: Embedding encoder for semantic similarity computations.
        limiter: Rate limiter for Semantic Scholar API requests.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        summ_prompt_key: Key for perspective summarisation prompt template.
        struct_prompt_key: Key for structured extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.
        sources: Set of paper sources to include in evaluation. If None, use all sources.
        perspectives: List of perspectives keys to evaluate. If None, use all available.
        filter_by_date: If True, filter recommended papers to only include those
            published before the main paper.
        callback: Optional callback function to call with phase names during processing.

    Returns:
        EvaluationResult with novelty evaluation and cost.

    Raises:
        ValueError: If paper is not found on arXiv or Semantic Scholar, or if
            processing fails at any stage.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    s2_api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    logger.debug("Processing paper: %s (arXiv:%s)", title, arxiv_id)

    paper = await atimer(
        get_paper_from_arxiv_id(arxiv_id, limiter, s2_api_key, callback=callback), 2
    )

    paper_annotated = await atimer(
        annotate_paper_pipeline(
            paper=paper,
            limiter=limiter,
            s2_api_key=s2_api_key,
            client=client,
            encoder=encoder,
            top_k_refs=top_k_refs,
            num_recommendations=num_recommendations,
            num_related=num_related,
            filter_by_date=filter_by_date,
            callback=callback,
        ),
        2,
    )

    graph_evaluated = await atimer(
        evaluate_paper_with_graph_multi(
            paper_annotated.result,
            client,
            eval_prompt_key,
            graph_prompt_key,
            summ_prompt_key,
            struct_prompt_key,
            demonstrations_key,
            demo_prompt_key,
            sources=sources,
            perspectives=perspectives,
            callback=callback,
        ),
        2,
    )

    return EvaluationResultMulti.from_(paper_annotated.then(graph_evaluated))
