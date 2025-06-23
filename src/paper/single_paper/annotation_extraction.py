"""GPT-based annotation extraction for papers and context classification.

This module handles extracting structured annotations from papers including:
- Key terms (methods and tasks)
- Background information extraction
- Target/contribution identification
- Citation context polarity (positive/negative)
"""

from __future__ import annotations

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
    """GPT-extracted annotated terms and classified abstract."""

    terms: gpt.PaperTerms
    abstr: GPTAbstractClassify


async def extract_annotations(
    title: str,
    abstract: str,
    client: LLMClient,
    term_prompt: gpt.PromptTemplate,
    abstract_prompt: gpt.PromptTemplate,
) -> GPTResult[PaperAnnotations]:
    """Extracting annotations from a paper given title and abstract.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        client: LLM client for GPT calls.
        term_prompt: Prompt template for term extraction.
        abstract_prompt: Prompt template for abstract classification.

    Returns:
        _PaperAnnotations (terms, abstract_classification) with fallbacks to empty
        values on error.
    """
    # Prepare prompts
    term_prompt_text = term_prompt.template.format(title=title, abstract=abstract)
    abstract_prompt_text = abstract_prompt.template.format(
        demonstrations="",  # TODO: No demonstrations for now
        abstract=abstract,
    )

    # Run GPT extractions
    result_term = await client.run(gpt.PaperTerms, TERM_SYSTEM_PROMPT, term_prompt_text)
    result_abstract = await client.run(
        GPTAbstractClassify, ABS_SYSTEM_PROMPT, abstract_prompt_text
    )

    # Extract results with fallbacks
    terms = result_term.result or gpt.PaperTerms.empty()
    abstract_classification = result_abstract.result or GPTAbstractClassify.empty()

    if not terms.is_valid():
        logger.warning("Paper '%s': invalid PaperTerms", title)
    if not abstract_classification.is_valid():
        logger.warning("Paper '%s': invalid GPTAbstractClassify", title)

    return GPTResult(
        result=PaperAnnotations(terms=terms, abstr=abstract_classification),
        cost=result_term.cost + result_abstract.cost,
    )


async def extract_paper_annotations(
    paper_with_s2_refs: s2.PaperWithS2Refs,
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[gpt.PeerReadAnnotated]:
    """Extract key terms and background/target information from paper using GPT."""
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
            background=ann.abstr.background,
            target=ann.abstr.target,
        )
    )


async def extract_recommended_annotations(
    recommended_papers: list[s2.Paper],
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[Sequence[gpt.PaperAnnotated]]:
    """Extract annotations from recommended papers using GPT."""
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
    """Extract terms and split abstract from paper."""
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
            background=a.abstr.background,
            target=a.abstr.target,
        )
    )


ReferenceIdx = NewType("ReferenceIdx", int)
"""Index of a reference from a paper. Used to split and group tasks."""


@dataclass(frozen=True, kw_only=True)
class ClassifyContextSpec:
    """Task data for classifying a single citation context."""

    main_title: str
    main_abstract: str
    reference: s2.S2Reference
    context: pr.CitationContext
    prompt: gpt.PromptTemplate


async def classify_context_single(
    client: LLMClient, spec: ClassifyContextSpec
) -> GPTResult[ContextClassified]:
    """Classify a single citation context using GPT."""
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
    """Classify all contexts concurrently and return results with reference indices."""
    tasks = [classify_context_single(client, task) for _, task in specs]
    results = await asyncio.gather(*tasks)
    return [(ref_idx, result) for (ref_idx, _), result in zip(specs, results)]


async def classify_citation_contexts(
    paper: s2.PaperWithS2Refs, client: LLMClient, context_prompt_key: str
) -> GPTResult[gpt.PaperWithContextClassfied]:
    """Classify citation contexts by polarity (positive/negative) using GPT."""
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
