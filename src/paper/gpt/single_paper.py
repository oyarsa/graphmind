"""Single paper processing pipeline for ORC dataset.

This module provides functionality to process a single paper through the complete PETER
pipeline, including S2 reference enhancement, GPT annotations, context classification,
related paper discovery, and summarization.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, NewType

import aiohttp
import arxiv  # type: ignore
import typer

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import related_papers as rp
from paper import semantic_scholar as s2
from paper.evaluation_metrics import TargetMode

# GPT: Term extraction, abstract splitting, context polarity classification, paper
# summarisation and graph evaluation.
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
from paper.gpt.evaluate_paper import GPTStructured, fix_evaluated_rating
from paper.gpt.evaluate_paper_graph import (
    GRAPH_EVAL_USER_PROMPTS,
    GRAPH_EXTRACT_USER_PROMPTS,
    format_eval_template,
    format_graph_template,
    get_demonstrations,
)
from paper.gpt.graph_types.full import GPTGraph
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_sequence
from paper.gpt.summarise_related_peter import (
    PETER_SUMMARISE_SYSTEM_PROMPT,
    PETER_SUMMARISE_USER_PROMPTS,
    GPTRelatedSummary,
    format_template,
)

# ORC: arXiv search, LaTeX download and parsing
from paper.orc.arxiv_api import (
    ArxivResult,
    arxiv_from_id,
    arxiv_id_from_url,
    arxiv_search,
    similar_titles,
)
from paper.orc.download import parse_arxiv_latex
from paper.orc.latex_parser import SentenceSplitter

# Semantic Scholar: paper metadata, citation information, recommended papers
from paper.semantic_scholar.info import (
    fetch_arxiv_papers,
    fetch_paper_data,
    fetch_paper_info,
    get_top_k_titles,
)
from paper.semantic_scholar.recommended import fetch_paper_recommendations

# Etc.
from paper.util import arun_safe, ensure_envvar, seqcat, setup_logging, timer
from paper.util.rate_limiter import Limiter, get_limiter

logger = logging.getLogger(__name__)


# Base S2 fields common to all queries
S2_FIELDS_BASE = [
    "paperId",
    "corpusId",
    "url",
    "title",
    "authors",
    "year",
    "abstract",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
]

# Full fields including venue and tldr
S2_FIELDS = [*S2_FIELDS_BASE, "tldr", "venue"]

REQUEST_TIMEOUT = 60  # 1 minute timeout for each request


async def get_paper_from_title(title: str, limiter: Limiter, api_key: str) -> pr.Paper:
    """Get a single processed Paper from a title using Semantic Scholar and arXiv.

    Args:
        title: Paper title to search for.
        limiter: Limiter for the requests.
        api_key: Semantic Scholar API key.

    Returns:
        Paper object with S2 metadata and parsed arXiv sections/references.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    # Run S2 and arXiv queries in parallel
    s2_paper, arxiv_result = await asyncio.gather(
        timer(
            fetch_s2_paper_info(api_key, title, limiter),
            ">>> fetch paper from s2",
        ),
        timer(get_arxiv_from_title(title), ">>> query arxiv from title"),
    )

    if not s2_paper:
        raise ValueError(f"Paper not found on Semantic Scholar: {title}")

    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {title}")
    logger.debug("arXiv result: %s", arxiv_result)

    # Parse arXiv LaTeX
    splitter = SentenceSplitter()
    sections, references = await timer(
        asyncio.to_thread(parse_arxiv_latex, arxiv_result, splitter),
        ">>> parse arXiv LaTeX",
    )

    # Create and return Paper object
    return pr.Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
    )


async def get_paper_from_arxiv_id(
    arxiv_id: str, limiter: Limiter, api_key: str
) -> pr.Paper:
    """Get a single processed Paper from the arXiv ID using Semantic Scholar and arXiv.

    Args:
        arxiv_id: ID of the paper on arXiv.
        limiter: Limiter for the requests.
        api_key: Semantic Scholar API key.

    Returns:
        Paper object with S2 metadata and parsed arXiv sections/references.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    # First get arXiv result to get the title for S2 lookup
    arxiv_result = await timer(get_arxiv_from_id(arxiv_id), ">>> get arxiv from id")
    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {arxiv_id}")
    logger.debug("arXiv result: %s", arxiv_result)

    # Run S2 lookup and LaTeX parsing in parallel
    s2_results, (sections, references) = await asyncio.gather(
        timer(
            fetch_arxiv_papers(
                api_key,
                [arxiv_result.arxiv_title],
                S2_FIELDS,
                desc="Fetching paper from S2",
                limiter=limiter,
            ),
            ">>> fetch paper from s2",
        ),
        timer(
            asyncio.to_thread(parse_arxiv_latex, arxiv_result, SentenceSplitter()),
            ">>> parse arXiv LaTeX",
        ),
    )

    if not s2_results or not s2_results[0]:
        raise ValueError(
            f"Paper not found on Semantic Scholar: {arxiv_result.arxiv_title}"
        )

    s2_paper = s2_results[0]

    # Create and return Paper object
    return pr.Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
    )


async def annotate_paper_pipeline(
    paper: pr.Paper,
    limiter: Limiter,
    s2_api_key: str,
    client: LLMClient,
    *,
    top_k_refs: int = 20,
    num_recommendations: int = 30,
    num_related: int = 2,
    term_prompt_key: str = "multi",
    abstract_prompt_key: str = "simple",
    context_prompt_key: str = "sentence",
    positive_prompt_key: str = "positive",
    negative_prompt_key: str = "negative",
    encoder_model: str = emb.DEFAULT_SENTENCE_MODEL,
    request_timeout: float = REQUEST_TIMEOUT,
) -> GPTResult[gpt.PaperWithRelatedSummary]:
    """Annotate a single paper through the complete pipeline to get related papers.

    Takes an ORC Paper and enhances it with:
    - S2 reference information for top-k semantically similar references
    - S2 recommended papers
    - GPT-extracted key terms, background, and target information
    - Citation context classification (positive/negative polarity)
    - PETER graph-based related paper discovery
    - GPT-generated summaries of related papers

    Args:
        paper: Base paper from ORC dataset with S2 metadata and arXiv
            sections/references.
        limiter: Limiter for the requests.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        term_prompt_key: Key for term annotation prompt template.
        abstract_prompt_key: Key for abstract classification prompt template.
        context_prompt_key: Key for context classification prompt template.
        positive_prompt_key: Key for positive paper summarisation prompt template.
        negative_prompt_key: Key for negative paper summarisation prompt template.
        encoder_model: Embedding encoder model.
        request_timeout: Maximum time (seconds) for S2 requests before timeout.
        s2_api_key: Semantic Scholar API key.
        client: LLM client for GPT API calls.

    Returns:
        Complete paper with related papers and their summaries wrapped in GPTResult.

    Raises:
        ValueError if no recommended papers or valid references were found.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    encoder = emb.Encoder(encoder_model)

    logger.debug("Processing paper: %s", paper.title)

    # Phase 1: Fetch S2 recommended papers and reference data in parallel
    logger.debug("Fetching S2 data for recommendations and references in parallel")
    paper_with_s2_refs, recommended_papers = await asyncio.gather(
        timer(
            enhance_with_s2_references(paper, top_k_refs, encoder, s2_api_key, limiter),
            ">>> enhance with s2 references",
        ),
        timer(
            fetch_s2_recommendations(
                paper, num_recommendations, s2_api_key, limiter, request_timeout
            ),
            ">>> fetch s2 recommendations",
        ),
    )

    if not paper_with_s2_refs or not recommended_papers:
        raise ValueError("No recommended found")

    # Phase 2: Extract annotations from all papers and classify contexts in parallel
    logger.debug("Processing all annotations and citation contexts in parallel")
    (
        paper_annotated,
        recommended_annotated,
        paper_with_classified_contexts,
    ) = await asyncio.gather(
        timer(
            extract_paper_annotations(
                paper_with_s2_refs, client, term_prompt_key, abstract_prompt_key
            ),
            ">>> extract paper annotations",
        ),
        timer(
            extract_recommended_annotations(
                recommended_papers, client, term_prompt_key, abstract_prompt_key
            ),
            ">>> extract recommended annotations",
        ),
        timer(
            classify_citation_contexts(paper_with_s2_refs, client, context_prompt_key),
            ">>> classify citation contexts",
        ),
    )

    # Phase 3: Get related papers (simplified approach without full PETER graphs)
    logger.debug("Getting related papers using direct approach")
    related_papers = await timer(
        asyncio.to_thread(
            get_related_papers,
            paper_annotated.result,
            paper_with_classified_contexts.result,
            recommended_annotated.result,
            num_related,
            encoder,
        ),
        ">>> get related papers direct",
    )

    # Phase 4: Generate summaries for related papers
    logger.debug("Generating summaries for related papers")
    related_papers_summarised = await timer(
        generate_related_paper_summaries(
            paper_annotated.result,
            related_papers,
            client,
            positive_prompt_key,
            negative_prompt_key,
        ),
        ">>> generate related paper summaries",
    )

    # Phase 5: Create final result
    logger.debug("Creating final result")
    result = gpt.PaperWithRelatedSummary(
        paper=paper_annotated.result, related=related_papers_summarised.result
    )

    total_cost = sum(
        x.cost
        for x in (
            paper_annotated,
            recommended_annotated,
            paper_with_classified_contexts,
            related_papers_summarised,
        )
    )
    return GPTResult(result=result, cost=total_cost)


async def enhance_with_s2_references(
    paper: pr.Paper, top_k: int, encoder: emb.Encoder, api_key: str, limiter: Limiter
) -> s2.PaperWithS2Refs:
    """Enhance paper with S2 reference information for top-k similar references."""
    # Get top-k reference titles by semantic similarity
    top_ref_titles = get_top_k_titles(encoder, paper, top_k)

    if not top_ref_titles:
        logger.warning("No references found for paper: %s", paper.title)
        # Return paper with empty S2 references
        return s2.PaperWithS2Refs.from_peer(paper, [])

    # Fetch S2 data for the top references
    s2_results = await fetch_arxiv_papers(
        api_key, top_ref_titles, [*S2_FIELDS_BASE, "tldr"], limiter=limiter
    )

    # Create S2Reference objects by matching with original references
    s2_papers_from_query = {
        s2.clean_title(paper.title_peer): paper for paper in s2_results if paper
    }

    s2_references = [
        s2.S2Reference.from_(s2_paper, contexts=ref.contexts)
        for ref in paper.references
        if (s2_paper := s2_papers_from_query.get(s2.clean_title(ref.title)))
    ]

    logger.debug(
        f"{len(paper.references) = } {len(s2_results) = } {len(s2_references) = }"
    )
    # Create enhanced paper with S2 references
    return s2.PaperWithS2Refs.from_peer(paper, s2_references)


@dataclass(frozen=True, kw_only=True)
class _PaperAnnotations:
    terms: gpt.PaperTerms
    abstr: GPTAbstractClassify


async def extract_annotations(
    title: str,
    abstract: str,
    client: LLMClient,
    term_prompt: gpt.PromptTemplate,
    abstract_prompt: gpt.PromptTemplate,
) -> GPTResult[_PaperAnnotations]:
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
        demonstrations="",  # No demonstrations for now
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
        result=_PaperAnnotations(terms=terms, abstr=abstract_classification),
        cost=result_term.cost + result_abstract.cost,
    )


async def extract_paper_annotations(
    paper_with_s2_refs: s2.PaperWithS2Refs,
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[gpt.PeerReadAnnotated]:
    """Extract key terms and background/target information from paper using GPT."""
    # Use specified prompts
    term_prompt = TERM_USER_PROMPTS[term_prompt_key]
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    # Extract annotations using shared helper
    result = await extract_annotations(
        paper_with_s2_refs.title,
        paper_with_s2_refs.abstract,
        client,
        term_prompt,
        abstract_prompt,
    )
    return result.map(
        lambda a: gpt.PeerReadAnnotated(
            terms=a.terms,
            paper=paper_with_s2_refs,
            background=a.abstr.background,
            target=a.abstr.target,
        )
    )


async def extract_recommended_annotations(
    recommended_papers: list[s2.Paper],
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> GPTResult[Sequence[gpt.PaperAnnotated]]:
    """Extract annotations from recommended papers using GPT."""
    # Use the same prompts as for the main paper
    term_prompt = TERM_USER_PROMPTS[term_prompt_key]
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    tasks = [
        extract_recommended_annotations_single(
            client, term_prompt, abstract_prompt, s2_paper
        )
        for s2_paper in recommended_papers
        if s2_paper.abstract
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


async def fetch_s2_recommendations(
    paper: pr.Paper,
    num_recommendations: int,
    api_key: str,
    limiter: Limiter,
    request_timeout: float,
) -> list[s2.Paper]:
    """Fetch recommended papers from S2 API for the given paper."""

    # First, we need to get the S2 paper ID by searching for this paper
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
    ) as session:
        data = await fetch_paper_data(
            session, api_key, paper.title, ["paperId"], limiter
        )

    if not data or not data.get("paperId"):
        logger.warning(
            "Paper '%s' not found on S2 - cannot fetch recommendations", paper.title
        )
        return []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(request_timeout), headers={"x-api-key": api_key}
    ) as session:
        return await fetch_paper_recommendations(
            session,
            paper.title,
            data["paperId"],
            S2_FIELDS_BASE,
            num_recommendations,
            limiter,
            from_="recent",
        )


@dataclass(frozen=True, kw_only=True)
class _ClassifyContextSpec:
    """Task data for classifying a single citation context."""

    reference: s2.S2Reference
    context: pr.CitationContext
    main_title: str
    main_abstract: str
    prompt: gpt.PromptTemplate


async def _classify_context_single(
    client: LLMClient, spec: _ClassifyContextSpec
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


_ReferenceIdx = NewType("_ReferenceIdx", int)
"""Index of a reference from a paper. Used to split and group tasks."""


async def _classify_contexts_parallel(
    client: LLMClient, specs: list[tuple[_ReferenceIdx, _ClassifyContextSpec]]
) -> list[tuple[_ReferenceIdx, GPTResult[ContextClassified]]]:
    """Classify all contexts in parallel and return results with reference indices."""
    tasks = [_classify_context_single(client, task) for _, task in specs]
    results = await asyncio.gather(*tasks)
    return [(ref_idx, result) for (ref_idx, _), result in zip(specs, results)]


async def classify_citation_contexts(
    paper: s2.PaperWithS2Refs, client: LLMClient, context_prompt_key: str
) -> GPTResult[gpt.PaperWithContextClassfied]:
    """Classify citation contexts by polarity (positive/negative) using GPT."""
    # Load prompts for context classification
    context_prompt = CONTEXT_USER_PROMPTS[context_prompt_key]

    # Create tasks for all contexts across all references
    tasks: list[tuple[_ReferenceIdx, _ClassifyContextSpec]] = []

    for ref_idx, reference in enumerate(paper.references):
        for context in reference.contexts:
            task = _ClassifyContextSpec(
                reference=reference,
                context=context,
                main_title=paper.title,
                main_abstract=paper.abstract,
                prompt=context_prompt,
            )
            tasks.append((_ReferenceIdx(ref_idx), task))

    classified_results = await _classify_contexts_parallel(client, tasks)

    # Group results by reference
    contexts_by_ref: dict[_ReferenceIdx, list[ContextClassified]] = defaultdict(list)
    total_cost = 0.0

    for ref_idx, result in classified_results:
        contexts_by_ref[ref_idx].append(result.result)
        total_cost += result.cost

    classified_references = [
        S2ReferenceClassified.from_(
            reference, contexts=contexts_by_ref[_ReferenceIdx(ref_idx)]
        )
        for ref_idx, reference in enumerate(paper.references)
    ]
    return GPTResult(
        result=PaperWithContextClassfied.from_(paper, classified_references),
        cost=total_cost,
    )


def get_related_papers(
    paper_annotated: gpt.PeerReadAnnotated,
    paper_with_contexts: gpt.PaperWithContextClassfied,
    recommended_papers: Sequence[gpt.PaperAnnotated],
    num_related: int,
    encoder: emb.Encoder,
) -> list[rp.PaperRelated]:
    """Get related papers using a direct approach without building full PETER graphs.

    For citations:
    - Get top `num_related` citations with positive and another `num_related` with
      negative polarity.

    For semantic:
    - Get recommended papers with extract terms (background/target).
    - Find top `num_related` papers by background similarity (negative polarity) and
      target similarity (positive polarity).

    This doesn't need a full PETER graph because we're only querying inside the citations
    and recommended papers, but the actual querying process is the same.
    """
    main_title_emb = encoder.encode(paper_annotated.title)
    main_background_emb = encoder.encode(paper_annotated.background)
    main_target_emb = encoder.encode(paper_annotated.target)

    references = paper_with_contexts.references
    references_positive_related = get_top_k_reference_by_polarity(
        encoder, main_title_emb, references, num_related, rp.ContextPolarity.POSITIVE
    )
    references_negative_related = get_top_k_reference_by_polarity(
        encoder, main_title_emb, references, num_related, rp.ContextPolarity.NEGATIVE
    )

    background_related = get_top_k_semantic(
        encoder,
        num_related,
        main_background_emb,
        recommended_papers,
        [r.background for r in recommended_papers],
        rp.ContextPolarity.NEGATIVE,
    )
    target_related = get_top_k_semantic(
        encoder,
        num_related,
        main_target_emb,
        recommended_papers,
        [r.target for r in recommended_papers],
        rp.ContextPolarity.POSITIVE,
    )

    logger.debug("Positive references: %d", len(references_positive_related))
    logger.debug("Negative references: %d", len(references_negative_related))
    logger.debug("Background related: %d", len(background_related))
    logger.debug("Target related: %d", len(target_related))

    return (
        references_positive_related
        + references_negative_related
        + background_related
        + target_related
    )


def get_top_k_semantic(
    encoder: emb.Encoder,
    k: int,
    main_emb: emb.Vector,
    papers: Sequence[gpt.PaperAnnotated],
    items: Sequence[str],
    polarity: rp.ContextPolarity,
) -> list[rp.PaperRelated]:
    """Get top K most similar papers by `items`."""
    sem_emb = encoder.encode_multi(items)
    sims = emb.similarities(main_emb, sem_emb)

    top_k_idx = emb.top_k_indices(sims, k)
    top_k = [(papers[i], float(sims[i])) for i in top_k_idx]

    return [
        rp.PaperRelated(
            source=rp.PaperSource.SEMANTIC,
            polarity=polarity,
            paper_id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            score=score,
        )
        for paper, score in top_k
    ]


def get_top_k_reference_by_polarity(
    encoder: emb.Encoder,
    title_emb: emb.Vector,
    references: Sequence[S2ReferenceClassified],
    k: int,
    polarity: rp.ContextPolarity,
) -> list[rp.PaperRelated]:
    """Get top K references by title similarity."""
    references_pol = [r for r in references if r.polarity.value == polarity.value]
    if not references_pol:
        return []

    titles_emb = encoder.encode_multi([r.title for r in references_pol])
    sims = emb.similarities(title_emb, titles_emb)

    top_k_idx = emb.top_k_indices(sims, k)
    top_k = [(references_pol[i], float(sims[i])) for i in top_k_idx]

    return [
        rp.PaperRelated(
            source=rp.PaperSource.CITATIONS,
            polarity=polarity,
            paper_id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            score=score,
        )
        for paper, score in top_k
    ]


async def generate_related_paper_summaries(
    paper_annotated: gpt.PeerReadAnnotated,
    related_papers: list[rp.PaperRelated],
    client: LLMClient,
    positive_prompt_key: str,
    negative_prompt_key: str,
) -> GPTResult[Sequence[gpt.PaperRelatedSummarised]]:
    """Generate GPT summaries for related papers."""
    if not related_papers:
        return GPTResult(result=[], cost=0)

    prompt_pol = {
        rp.ContextPolarity.POSITIVE: PETER_SUMMARISE_USER_PROMPTS[positive_prompt_key],
        rp.ContextPolarity.NEGATIVE: PETER_SUMMARISE_USER_PROMPTS[negative_prompt_key],
    }

    # Create tasks for parallel execution
    tasks = [
        generate_summary_single(
            paper_annotated, client, prompt_pol[related_paper.polarity], related_paper
        )
        for related_paper in related_papers
    ]
    return gpt_sequence(await asyncio.gather(*tasks))


async def generate_summary_single(
    paper_annotated: gpt.PeerReadAnnotated,
    client: LLMClient,
    user_prompt: gpt.PromptTemplate,
    related_paper: rp.PaperRelated,
) -> GPTResult[gpt.PaperRelatedSummarised]:
    """Generate a single summary for a related paper."""
    # Format prompt using the same function from summarise_related_peter.py
    user_prompt_text = format_template(
        user_prompt,
        paper_annotated,
        related_paper,
    )

    result = await client.run(
        GPTRelatedSummary, PETER_SUMMARISE_SYSTEM_PROMPT, user_prompt_text
    )
    return result.map(
        lambda r: gpt.PaperRelatedSummarised.from_related(
            related_paper, (r or GPTRelatedSummary.error()).summary
        )
    )


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
) -> GPTResult[gpt.GraphResult]:
    """Evaluate a paper's novelty using graph extraction and related papers.

    Args:
        paper: Paper with related papers and summaries from PETER pipeline.
        client: LLM client for GPT API calls.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.

    Returns:
        GraphResult with novelty evaluation wrapped in GPTResult.
    """
    eval_prompt, graph_prompt = get_prompts(eval_prompt_key, graph_prompt_key)
    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    graph_result = await timer(
        extract_graph_from_paper(paper.paper, client, graph_prompt),
        ">>> extract graph",
    )

    eval_result = await timer(
        evaluate_paper_graph_novelty(
            paper, graph_result.result, client, eval_prompt, demonstrations
        ),
        ">>> evaluate graph",
    )

    return GPTResult(
        result=construct_graph_result(paper, graph_result.result, eval_result.result),
        cost=graph_result.cost + eval_result.cost,
    )


class QueryType(StrEnum):
    """Whether to query arXiv by title or ID/URL."""

    TITLE = "title"
    ID = "id"


def single_paper(
    query: Annotated[
        str, typer.Argument(help="Title or arXiv ID/URL of the paper to process")
    ],
    type_: Annotated[
        QueryType, typer.Option("--type", help="Whether to query by title or arXiv.")
    ],
    top_k_refs: Annotated[
        int, typer.Option(help="Number of top references to process by similarity")
    ] = 20,
    num_recommendations: Annotated[
        int, typer.Option(help="Number of recommended papers to fetch from S2 API")
    ] = 30,
    num_related: Annotated[
        int, typer.Option(help="Number of related papers per type (positive/negative)")
    ] = 2,
    llm_model: Annotated[
        str, typer.Option(help="GPT/Gemini model to use for API calls")
    ] = "gpt-4o-mini",
    encoder_model: Annotated[
        str, typer.Option(help="Embedding encoder model")
    ] = emb.DEFAULT_SENTENCE_MODEL,
    seed: Annotated[int, typer.Option(help="Random seed for GPT API calls")] = 0,
    detail: Annotated[
        bool, typer.Option(help="Show detailed paper information.")
    ] = False,
    eval_prompt: Annotated[
        str, typer.Option(help="User prompt for paper evaluation")
    ] = "full-graph-structured",
    graph_prompt: Annotated[
        str, typer.Option(help="User prompt for graph extraction")
    ] = "full",
    demonstrations: Annotated[
        str, typer.Option(help="Demonstrations file for few-shot prompting")
    ] = "orc_4",
    demo_prompt: Annotated[
        str, typer.Option(help="Demonstration prompt key")
    ] = "abstract",
) -> None:
    """Process a paper title through the complete PETER pipeline and print results."""
    setup_logging()
    arun_safe(
        process_paper,
        query,
        type_,
        top_k_refs,
        num_recommendations,
        num_related,
        llm_model,
        encoder_model,
        seed,
        detail,
        eval_prompt,
        graph_prompt,
        demonstrations,
        demo_prompt,
    )


async def process_paper_from_query(
    query: str,
    type_: QueryType,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
    limiter: Limiter,
    eval_prompt_key: str,
    graph_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
) -> GPTResult[gpt.GraphResult]:
    """Process a paper by query (title or arXiv ID/URL) through the complete PETER pipeline.

    This function provides a complete end-to-end paper processing pipeline that:
    1. Retrieves paper from Semantic Scholar and arXiv using the title or ID/URL
    2. Processes the paper through the complete PETER pipeline including:
       - S2 reference enhancement with top-k semantic similarity filtering
       - GPT-based annotation extraction (key terms, background, target)
       - Citation context classification (positive/negative polarity)
       - Related paper discovery via citations and semantic matching
       - GPT-generated summaries of related papers
    3. Extracts a graph representation and evaluates novelty

    Args:
        query: Paper title or arXiv ID/URL to search for and process.
        type_: Whether to query arXiv by title or ID/URL.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        llm_model: GPT/Gemini model to use for all LLM API calls.
        encoder_model: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        limiter: Rate limiter for Semantic Scholar API requests to prevent 429 errors.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.

    Returns:
        GraphResult with novelty evaluation wrapped in GPTResult.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv, or if no
            recommended papers or valid references are found during processing.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    s2_api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    client = LLMClient.new_env(llm_model, seed)

    match type_:
        case QueryType.TITLE:
            logger.debug("Searching by title: %s", query)
            paper = await timer(
                get_paper_from_title(query, limiter, s2_api_key),
                ">> get paper from title",
            )
        case QueryType.ID:
            arxiv_id = arxiv_id_from_url(query)
            logger.debug("Searching by arXiv ID: %s", arxiv_id)
            paper = await timer(
                get_paper_from_arxiv_id(arxiv_id, limiter, s2_api_key),
                ">> get paper from arxiv ID",
            )

    # Process paper through PETER pipeline
    paper_result = await timer(
        annotate_paper_pipeline(
            paper,
            limiter,
            s2_api_key,
            client,
            top_k_refs=top_k_refs,
            num_recommendations=num_recommendations,
            num_related=num_related,
            encoder_model=encoder_model,
        ),
        ">> annotate paper pipeline",
    )

    # Evaluate with graph
    graph_result = await timer(
        evaluate_paper_with_graph(
            paper_result.result,
            client,
            eval_prompt_key,
            graph_prompt_key,
            demonstrations_key,
            demo_prompt_key,
        ),
        ">> evaluate paper with graph",
    )

    return paper_result.then(graph_result)


def display_graph_results(result: gpt.GraphResult) -> None:
    """Display comprehensive results from processed paper with graph.

    Prints a formatted summary of the paper processing results including:
    - Main paper details (title, abstract, key terms, background, target)
    - Novelty evaluation results
    - Graph information
    - Citation-based related papers (positive and negative)
    - Semantic-based related papers (positive and negative)
    - Summaries and scores for each related paper

    Args:
        result: Complete processed paper with graph and evaluation.
    """
    print("\n" + "=" * 80)
    print("📊 COMPLETE PAPER PROCESSING RESULTS")
    print("=" * 80)

    # Print main paper information
    print(f"📑 Title: {result.paper.title}")
    print(f"📝 Abstract: {result.paper.abstract}")
    if result.terms:
        key_terms = seqcat(result.terms.methods, result.terms.tasks)
        print(f"🏷️  Key Terms: {', '.join(key_terms)}")
    if result.background:
        print(f"🎯 Background: {result.background}")
    if result.target:
        print(f"🚀 Target: {result.target}")
    print(f"📊 S2 References: {len(result.paper.references)}")
    print()

    # Print related papers summary
    if result.related:
        print(f"🔗 RELATED PAPERS ({len(result.related)} total):")
        print("-" * 50)
    else:
        print("🔗 RELATED PAPERS (0 total):")
        print("-" * 50)

    print("\n📖 CITATION-BASED PAPERS:")

    print("\n  ✅ Positive citations:")
    citations_positive = filter_related(
        result, rp.ContextPolarity.POSITIVE, rp.PaperSource.CITATIONS
    )
    print("\n".join(map(display_related_paper, citations_positive)))

    print("\n  ❌ Negative citations:")
    citations_negative = filter_related(
        result, rp.ContextPolarity.NEGATIVE, rp.PaperSource.CITATIONS
    )
    print("\n".join(map(display_related_paper, citations_negative)))

    print("\n🔍 SEMANTIC-BASED PAPERS:")

    print("\n  ✅ Positive semantic matches:")
    semantic_positive = filter_related(
        result, rp.ContextPolarity.POSITIVE, rp.PaperSource.SEMANTIC
    )
    print("\n".join(map(display_related_paper, semantic_positive)))

    print("\n  ❌ Negative semantic matches:")
    semantic_negative = filter_related(
        result, rp.ContextPolarity.NEGATIVE, rp.PaperSource.SEMANTIC
    )
    print("\n".join(map(display_related_paper, semantic_negative)))


async def process_paper(
    query: str,
    type_: QueryType,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
    detail: bool,
    eval_prompt: str,
    graph_prompt: str,
    demonstrations: str,
    demo_prompt: str,
) -> None:
    """Process a paper by title and display results.

    Convenience function that combines processing and display:
    1. Retrieves and processes paper through complete PETER pipeline.
    2. Displays comprehensive results to stdout.

    Args:
        query: Paper title or arXiv ID/URL to search for and process.
        type_: Whether to query arXiv by title or ID/URL.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        llm_model: GPT/Gemini model to use for all LLM API calls.
        encoder_model: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        detail: Show detailed paper information.
        eval_prompt: User prompt for paper evaluation.
        graph_prompt: User prompt for graph extraction.
        demonstrations: Demonstrations file for few-shot prompting.
        demo_prompt: Demonstration prompt key.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv, or if no
            recommended papers or valid references are found during processing.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    limiter = get_limiter(1, 1)  # 1 request per second

    result = await timer(
        process_paper_from_query(
            query,
            type_,
            top_k_refs,
            num_recommendations,
            num_related,
            llm_model,
            encoder_model,
            seed,
            limiter,
            eval_prompt,
            graph_prompt,
            demonstrations,
            demo_prompt,
        ),
        "> process paper from query",
    )

    graph_result = result.result
    print(f"✅ Found paper: {graph_result.paper.title}")
    print(f"📄 Abstract: {graph_result.paper.abstract[:200]}...")
    print(f"📚 References: {len(graph_result.paper.references)}")
    print(f"📖 Sections: {len(graph_result.paper.sections)}")
    print()
    print("🚀 Processing through PETER pipeline and graph evaluation...")
    print(f"💰 Total cost: ${result.cost:.10f}")

    # Display novelty evaluation
    print(f"\n🎯 Novelty Evaluation: {graph_result.paper.label}")
    print(f"📝 Rationale: {graph_result.paper.rationale_pred[:1000]}")

    if graph_result.graph and not graph_result.graph.is_empty():
        print(f"\n📊 Graph extracted with {len(graph_result.graph.entities)} entities")

    if detail:
        display_graph_results(graph_result)


def filter_related(
    result: gpt.GraphResult, pol: rp.ContextPolarity, src: rp.PaperSource
) -> list[gpt.PaperRelatedSummarised]:
    """Filter related papers by polarity and source."""
    if not result.related:
        return []

    return [
        r
        for r in result.related
        if r.source.value == src.value and r.polarity.value == pol.value
    ]


def display_related_paper(related: gpt.PaperRelatedSummarised) -> str:
    """Display summary of related paper."""
    out = [
        f"    • {related.title}",
        f"      Score: {related.score:.3f}",
        f"      Summary: {related.summary[:100]}...",
    ]
    return "\n".join(out) + "\n"


async def get_arxiv_from_title(openreview_title: str) -> ArxivResult | None:
    """Get ArxivResult for a single OpenReview paper title if it exists on arXiv."""
    client = arxiv.Client()

    query = f'ti:"{openreview_title}"'

    try:
        result = next(await asyncio.to_thread(arxiv_search, client, query, 1))
        if similar_titles(openreview_title, result.title):
            return ArxivResult(
                id=arxiv_id_from_url(result.entry_id),
                openreview_title=openreview_title,
                arxiv_title=result.title,
            )
    except Exception as e:
        logger.warning(f"Error searching for '{openreview_title}' on arXiv: {e}")

    return None


async def get_arxiv_from_id(arxiv_id: str) -> ArxivResult | None:
    """Get ArxivResult for a single OpenReview paper title if it exists on arXiv."""
    client = arxiv.Client()

    try:
        result = next(await asyncio.to_thread(arxiv_from_id, client, arxiv_id))
        return ArxivResult(
            id=arxiv_id,
            openreview_title=result.title,
            arxiv_title=result.title,
        )
    except Exception as e:
        logger.warning(f"Error searching for '{arxiv_id}' on arXiv: {e}")

    return None


async def fetch_s2_paper_info(
    api_key: str, title: str, limiter: Limiter
) -> s2.PaperFromPeerRead | None:
    """Fetch paper information from the Semantic Scholar API.

    Args:
        api_key: Semantic Scholar API key.
        title: Paper title to search for.
        limiter: Limiter for the API requests.

    Returns:
        Paper found or None for failed/not found papers.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT)
    ) as session:
        return await fetch_paper_info(session, api_key, title, S2_FIELDS, limiter)
