"""GPT-based annotation extraction for papers and citation context classification.

This module provides functionality to extract structured information from academic papers
using Large Language Models (GPT). It handles two main types of annotation tasks:

1. Paper Annotation Extraction:
- Key terms extraction: Identifies important methods and tasks mentioned in papers.
- Background/target classification: Splits background context from paper contributions.
- Works with both individual papers and lists of recommended papers.

2. Citation Context Classification:
- Analyses citation contexts to determine their polarity (positive/negative).
- Helps understand how papers reference each other (supportive vs critical).
- Groups classification results by reference for easy analysis.

The module integrates with the Semantic Scholar API for paper data and uses predefined
prompts for consistent GPT interactions. All operations are async for
efficient concurrent processing of multiple papers or contexts.

Key Features:
- Graceful error handling with fallback to empty values.
- Cost tracking for GPT API usage.
- Support for batch processing of multiple papers.
- Validation of extracted annotations.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType

from paper import gpt
from paper import peerread as pr
from paper import semantic_scholar as s2
from paper.gpt.annotate_paper import (
    ABS_SYSTEM_PROMPT,
    ABS_USER_PROMPTS,
    TERM_SYSTEM_PROMPT,
    TERM_USER_PROMPTS,
    GPTAbstractClassify,
)
from paper.gpt.classify_contexts import (
    CONTEXT_SYSTEM_PROMPT,
    CONTEXT_USER_PROMPTS,
    ContextClassified,
    GPTContext,
    PaperWithContextClassfied,
    S2ReferenceClassified,
)
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PaperAnnotations:
    """Container for GPT-extracted paper annotations.

    Combines term extraction results with abstract classification to provide a complete
    annotation of a paper's key contributions and context.

    Attributes:
        terms: Extracted key terms including methods and tasks.
        abstract: Classification of abstract into background and target sections.
    """

    terms: gpt.PaperTerms
    abstract: GPTAbstractClassify


async def extract_annotations(
    title: str,
    abstract: str,
    client: LLMClient,
    term_prompt: gpt.PromptTemplate,
    abstract_prompt: gpt.PromptTemplate,
) -> GPTResult[PaperAnnotations]:
    """Extract comprehensive annotations from a paper using GPT.

    Performs parallel extraction of key terms and abstract classification, combining
    results into a single annotation object. Handles errors gracefully by falling back
    to empty values.

    Args:
        title: Paper title for extraction context.
        abstract: Paper abstract to analyse.
        client: LLM client for GPT calls.
        term_prompt: Template for term extraction prompts.
        abstract_prompt: Template for abstract classification prompts.

    Returns:
        GPTResult containing PaperAnnotations with both term extraction and abstract
        classification results, along with total API cost.
    """
    # Prepare prompts
    term_prompt_text = term_prompt.template.format(title=title, abstract=abstract)
    abstract_prompt_text = abstract_prompt.template.format(
        demonstrations="",
        abstract=abstract,
    )

    # Run GPT extractions
    result_term, result_abstract = await asyncio.gather(
        client.run(gpt.PaperTerms, TERM_SYSTEM_PROMPT, term_prompt_text),
        client.run(GPTAbstractClassify, ABS_SYSTEM_PROMPT, abstract_prompt_text),
    )

    # Extract results with fallbacks
    terms = result_term.result or gpt.PaperTerms.empty()
    abstract_classification = result_abstract.result or GPTAbstractClassify.empty()

    if not terms.is_valid():
        logger.warning("Paper '%s': invalid PaperTerms", title)
    if not abstract_classification.is_valid():
        logger.warning("Paper '%s': invalid GPTAbstractClassify", title)

    return GPTResult(
        result=PaperAnnotations(terms=terms, abstract=abstract_classification),
        cost=result_term.cost + result_abstract.cost,
    )


async def extract_main_paper_annotations(
    paper_with_s2_refs: s2.PaperWithS2Refs,
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[gpt.PeerReadAnnotated]:
    """Extract annotations from a PeerRead paper with Semantic Scholar references.

    This is the main entry point for annotating individual papers in the PeerRead
    dataset. It extracts both key terms and classifies the abstract into background
    and target sections.

    Args:
        paper_with_s2_refs: Paper data including S2 references.
        client: LLM client for GPT calls.
        term_prompt_key: Key to select term extraction prompt from TERM_USER_PROMPTS.
        abstract_prompt_key: Key to select abstract prompt from ABS_USER_PROMPTS.

    Returns:
        GPTResult containing PeerReadAnnotated with extracted annotations and costs.
    """
    term_prompt = TERM_USER_PROMPTS[term_prompt_key]
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    result = await extract_annotations(
        paper_with_s2_refs.title,
        paper_with_s2_refs.abstract,
        client,
        term_prompt,
        abstract_prompt,
    )
    return result.map(
        lambda ann: gpt.PeerReadAnnotated(
            terms=ann.terms,
            paper=paper_with_s2_refs,
            background=ann.abstract.background,
            target=ann.abstract.target,
        )
    )


async def extract_recommended_annotations(
    recommended_papers: list[s2.Paper],
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[Sequence[gpt.PaperAnnotated]]:
    """Extract annotations from a list of recommended papers concurrently.

    Processes multiple papers in parallel for efficiency. Only processes papers that
    have both title and abstract. Aggregates costs across all extractions.

    Args:
        recommended_papers: List of S2 papers to annotate.
        client: LLM client for GPT calls.
        term_prompt_key: Key for term extraction prompt selection.
        abstract_prompt_key: Key for abstract classification prompt.

    Returns:
        GPTResult with sequence of annotated papers and total cost.
    """
    term_prompt = TERM_USER_PROMPTS[term_prompt_key]
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    tasks = [
        extract_recommended_annotations_single(
            client, term_prompt, abstract_prompt, s2_paper
        )
        for s2_paper in recommended_papers
        if s2_paper.abstract and s2_paper.title
    ]
    return gpt_sequence(await asyncio.gather(*tasks))


async def extract_recommended_annotations_single(
    client: LLMClient,
    term_prompt: gpt.PromptTemplate,
    abstract_prompt: gpt.PromptTemplate,
    s2_paper: s2.Paper,
) -> GPTResult[gpt.PaperAnnotated]:
    """Extract annotations from a single Semantic Scholar paper.

    Helper function that processes individual papers from the recommended list.
    Ensures the paper has required fields before processing.

    Args:
        client: LLM client for GPT calls
        term_prompt: Template for term extraction prompts
        abstract_prompt: Template for abstract classification
        s2_paper: Individual S2 paper to process

    Returns:
        GPTResult containing PaperAnnotated with all extracted information
    """
    # We know these are not None because of the filter in extract_recommended_annotations
    assert s2_paper.title is not None
    assert s2_paper.abstract is not None

    # Extract annotations using shared helper
    result = await extract_annotations(
        s2_paper.title,
        s2_paper.abstract,
        client,
        term_prompt,
        abstract_prompt,
    )
    return result.map(
        lambda a: gpt.PaperAnnotated(
            terms=a.terms,
            paper=s2_paper,
            background=a.abstract.background,
            target=a.abstract.target,
        )
    )


ReferenceIdx = NewType("ReferenceIdx", int)
"""Type-safe index for paper references.

Used to maintain the relationship between citation contexts and their
corresponding references when processing contexts in parallel."""


@dataclass(frozen=True, kw_only=True)
class ClassifyContextSpec:
    """Specification for classifying a single citation context.

    Contains all necessary information to classify how a paper cites another,
    including both papers' metadata and the citation context itself.

    Attributes:
        main_title: Title of the citing paper
        main_abstract: Abstract of the citing paper
        reference: The cited paper's information
        context: The citation context to classify
        prompt: Template for generating the classification prompt
    """

    main_title: str
    main_abstract: str
    reference: s2.S2Reference
    context: pr.CitationContext
    prompt: gpt.PromptTemplate


async def classify_context_single(
    client: LLMClient, spec: ClassifyContextSpec
) -> GPTResult[ContextClassified]:
    """Classify the polarity of a single citation context.

    Determines whether a citation is positive (supportive) or negative (critical)
    based on the context sentence and both papers' content. Falls back to
    positive classification if GPT fails.

    Args:
        client: LLM client for GPT calls
        spec: Classification task specification

    Returns:
        GPTResult with classified context including prediction and gold label
    """
    user_prompt_text = spec.prompt.template.format(
        main_title=spec.main_title,
        main_abstract=spec.main_abstract,
        reference_title=spec.reference.title,
        reference_abstract=spec.reference.abstract,
        context=spec.context.sentence,
    )

    result = await client.run(GPTContext, CONTEXT_SYSTEM_PROMPT, user_prompt_text)
    # Fallback to positive if GPT fails
    return result.map(
        lambda r: ContextClassified(
            text=spec.context.sentence,
            gold=spec.context.polarity,
            prediction=r.polarity if r else pr.ContextPolarity.POSITIVE,
        )
    )


async def classify_contexts(
    client: LLMClient, specs: list[tuple[ReferenceIdx, ClassifyContextSpec]]
) -> list[tuple[ReferenceIdx, GPTResult[ContextClassified]]]:
    """Classify multiple citation contexts in parallel.

    Processes all classification tasks concurrently for efficiency while
    maintaining the association between contexts and their references through
    the ReferenceIdx.

    Args:
        client: LLM client for GPT calls
        specs: List of (index, specification) tuples to process

    Returns:
        List of (index, result) tuples maintaining original associations
    """
    tasks = [classify_context_single(client, task) for _, task in specs]
    results = await asyncio.gather(*tasks)
    return [(ref_idx, result) for (ref_idx, _), result in zip(specs, results)]


async def classify_citation_contexts(
    paper: s2.PaperWithS2Refs, client: LLMClient, context_prompt_key: str
) -> GPTResult[gpt.PaperWithContextClassfied]:
    """Classify all citation contexts in a paper by polarity.

    Main entry point for citation context classification. Processes all contexts
    for all references in a paper, grouping results by reference for easy analysis.
    Tracks total GPT API costs across all classifications.

    Args:
        paper: Paper containing references with citation contexts
        client: LLM client for GPT calls
        context_prompt_key: Key to select prompt from CONTEXT_USER_PROMPTS

    Returns:
        GPTResult with PaperWithContextClassfied containing all classified contexts
        organised by reference, along with total API cost
    """
    context_prompt = CONTEXT_USER_PROMPTS[context_prompt_key]

    tasks: list[tuple[ReferenceIdx, ClassifyContextSpec]] = []

    for ref_idx, reference in enumerate(paper.references):
        for context in reference.contexts:
            task = ClassifyContextSpec(
                reference=reference,
                context=context,
                main_title=paper.title,
                main_abstract=paper.abstract,
                prompt=context_prompt,
            )
            tasks.append((ReferenceIdx(ref_idx), task))

    classified_results = await classify_contexts(client, tasks)

    # Group results by reference
    contexts_by_ref: dict[ReferenceIdx, list[ContextClassified]] = defaultdict(list)
    total_cost = 0.0

    for ref_idx, result in classified_results:
        contexts_by_ref[ref_idx].append(result.result)
        total_cost += result.cost

    classified_references = [
        S2ReferenceClassified.from_(
            reference, contexts=contexts_by_ref[ReferenceIdx(ref_idx)]
        )
        for ref_idx, reference in enumerate(paper.references)
    ]
    return GPTResult(
        result=PaperWithContextClassfied.from_(paper, classified_references),
        cost=total_cost,
    )
