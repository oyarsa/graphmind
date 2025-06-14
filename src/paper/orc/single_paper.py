"""Single paper processing pipeline for ORC dataset.

This module provides functionality to process a single paper through the complete PETER
pipeline, including S2 reference enhancement, GPT annotations, context classification,
related paper discovery, and summarization.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, NewType

import aiohttp
import typer

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import related_papers as rp
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
from paper.gpt.summarise_related_peter import (
    PETER_SUMMARISE_SYSTEM_PROMPT,
    PETER_SUMMARISE_USER_PROMPTS,
    GPTRelatedSummary,
    format_template,
)
from paper.orc.arxiv_api import get_arxiv, normalise_title
from paper.orc.download import parse_arxiv_latex
from paper.orc.latex_parser import SentenceSplitter
from paper.peerread.model import Paper
from paper.semantic_scholar.info import (
    fetch_arxiv_papers,
    fetch_paper_data,
    get_top_k_titles,
)
from paper.semantic_scholar.model import Paper as S2Paper
from paper.semantic_scholar.model import PaperWithS2Refs
from paper.semantic_scholar.recommended import fetch_paper_recommendations
from paper.util import Timer, arun_safe, ensure_envvar, progress
from paper.util.rate_limiter import Limiter, get_limiter

if TYPE_CHECKING:
    from paper.gpt.model import PaperRelatedSummarised
    from paper.gpt.prompts import PromptTemplate

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


async def get_paper_from_title(title: str, limiter: Limiter, api_key: str) -> Paper:
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
    s2_results = await timer(
        fetch_arxiv_papers(
            api_key, [title], S2_FIELDS, desc="Fetching paper from S2", limiter=limiter
        ),
        "fetch paper from s2",
    )

    if not s2_results or not s2_results[0]:
        raise ValueError(f"Paper not found on Semantic Scholar: {title}")

    s2_paper = s2_results[0]

    # Query arXiv for LaTeX content
    openreview_to_arxiv = await timer(
        asyncio.to_thread(get_arxiv, [s2_paper.title], batch_size=1),
        "query arxiv for latex",
    )

    normalized_title = normalise_title(s2_paper.title)
    arxiv_result = openreview_to_arxiv.get(normalized_title)

    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {s2_paper.title}")

    # Parse arXiv LaTeX
    splitter = SentenceSplitter()
    sections, references = await timer(
        asyncio.to_thread(parse_arxiv_latex, arxiv_result, splitter),
        "parse arXiv LaTeX",
    )

    # Create and return Paper object
    return Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
    )


async def process_paper_complete(
    paper: Paper,
    limiter: Limiter,
    s2_api_key: str,
    *,
    top_k_refs: int = 20,
    num_recommendations: int = 30,
    num_related: int = 2,
    term_prompt_key: str = "multi",
    abstract_prompt_key: str = "simple",
    context_prompt_key: str = "sentence",
    positive_prompt_key: str = "positive",
    negative_prompt_key: str = "negative",
    llm_model: str = "gpt-4o-mini",
    encoder_model: str = emb.DEFAULT_SENTENCE_MODEL,
    seed: int = 0,
    request_timeout: float = REQUEST_TIMEOUT,
) -> gpt.PaperWithRelatedSummary:
    """Process a single paper through the complete pipeline to get related papers.

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
        llm_model: GPT/Gemini model to use for all API calls.
        encoder_model: Embedding encoder model.
        seed: Random seed for GPT API calls.
        request_timeout: Maximum time (seconds) for S2 requests before timeout.
        s2_api_key: Semantic Scholar API key.

    Returns:
        Complete paper with related papers and their summaries.

    Raises:
        ValueError if no recommended papers or valid references were found.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    # Initialize shared resources
    client = LLMClient.new_env(llm_model, seed)
    encoder = emb.Encoder(encoder_model)

    logger.info("Processing paper: %s", paper.title)

    # Phase 1: Fetch S2 recommended papers and reference data in parallel
    logger.debug("Fetching S2 data for recommendations and references in parallel")
    paper_with_s2_refs, recommended_papers = await asyncio.gather(
        timer(
            enhance_with_s2_references(paper, top_k_refs, encoder, s2_api_key, limiter),
            "enhance with s2 references",
        ),
        timer(
            fetch_s2_recommendations(
                paper, num_recommendations, s2_api_key, limiter, request_timeout
            ),
            "fetch s2 recommendations",
        ),
    )

    if not paper_with_s2_refs or not recommended_papers:
        raise ValueError("No recommended found")

    # Phase 2: Extract annotations from main paper
    logger.debug("Extracting key terms and background/target of main paper via GPT")
    paper_annotated = await timer(
        extract_paper_annotations(
            paper_with_s2_refs, client, term_prompt_key, abstract_prompt_key
        ),
        "extract paper annotations",
    )

    # Phase 3: Extract annotations from recommended papers and classify contexts in parallel
    logger.debug("Processing recommended papers and citation contexts in parallel")
    recommended_annotated, paper_with_classified_contexts = await asyncio.gather(
        timer(
            extract_recommended_annotations(
                recommended_papers, client, term_prompt_key, abstract_prompt_key
            ),
            "extract recommended annotations",
        ),
        timer(
            classify_citation_contexts(paper_with_s2_refs, client, context_prompt_key),
            "classify citation contexts",
        ),
    )

    # Phase 4: Get related papers (simplified approach without full PETER graphs)
    logger.debug("Getting related papers using direct approach")
    related_papers = await timer(
        get_related_papers_direct(
            paper_annotated.result,
            paper_with_classified_contexts.result,
            recommended_annotated.result,
            num_related,
            encoder,
        ),
        "get related papers direct",
    )

    # Phase 5: Generate summaries for related papers
    logger.debug("Generating summaries for related papers")
    related_papers_summarised = await timer(
        generate_related_paper_summaries(
            paper_annotated.result,
            related_papers,
            client,
            positive_prompt_key,
            negative_prompt_key,
        ),
        "generate related paper summaries",
    )

    # Phase 6: Create final result
    logger.debug("Creating final result")
    result = gpt.PaperWithRelatedSummary(
        paper=paper_annotated.result, related=related_papers_summarised.result
    )

    logger.info("Completed processing paper: %s", paper.title)
    total_cost = sum(
        x.cost
        for x in (
            paper_annotated,
            recommended_annotated,
            paper_with_classified_contexts,
            related_papers_summarised,
        )
    )
    logger.info("Cost: %f", total_cost)
    return result


async def timer[T](task: Awaitable[T], name: str) -> T:
    """Print time it takes to run task."""
    with Timer(name) as t:
        result = await task
    logger.warning(t)
    return result


async def enhance_with_s2_references(
    paper: Paper, top_k: int, encoder: emb.Encoder, api_key: str, limiter: Limiter
) -> PaperWithS2Refs:
    """Enhance paper with S2 reference information for top-k similar references."""
    # Get top-k reference titles by semantic similarity
    top_ref_titles = get_top_k_titles(encoder, paper, top_k)

    if not top_ref_titles:
        logger.warning("No references found for paper: %s", paper.title)
        # Return paper with empty S2 references
        return PaperWithS2Refs.from_peer(paper, [])

    # Fetch S2 data for the top references
    s2_results = await fetch_arxiv_papers(
        api_key,
        top_ref_titles,
        [*S2_FIELDS_BASE, "tldr"],
        desc="Fetching S2 data for references",
        limiter=limiter,
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
    return PaperWithS2Refs.from_peer(paper, s2_references)


@dataclass(frozen=True, kw_only=True)
class _PaperAnnotations:
    terms: gpt.PaperTerms
    abstr: GPTAbstractClassify


async def extract_annotations(
    title: str,
    abstract: str,
    client: LLMClient,
    term_prompt: PromptTemplate,
    abstract_prompt: PromptTemplate,
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
    paper_with_s2_refs: PaperWithS2Refs,
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
    recommended_papers: list[S2Paper],
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
    return gpt_sequence(await progress.gather(tasks, desc="Annotating related papers."))


async def extract_recommended_annotations_single(
    client: LLMClient,
    term_prompt: PromptTemplate,
    abstract_prompt: PromptTemplate,
    s2_paper: S2Paper,
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
    paper: Paper,
    num_recommendations: int,
    api_key: str,
    limiter: Limiter,
    request_timeout: float,
) -> list[S2Paper]:
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
    prompt: PromptTemplate


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
    results = await progress.gather(tasks, desc="Classifying contexts")
    return [(ref_idx, result) for (ref_idx, _), result in zip(specs, results)]


async def classify_citation_contexts(
    paper: PaperWithS2Refs, client: LLMClient, context_prompt_key: str
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


async def get_related_papers_direct(
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

    backgrounds = [r.background for r in recommended_papers]
    background_related = get_top_k_semantic(
        encoder,
        num_related,
        main_background_emb,
        recommended_papers,
        backgrounds,
        rp.ContextPolarity.POSITIVE,
    )
    targets = [r.target for r in recommended_papers]
    target_related = get_top_k_semantic(
        encoder,
        num_related,
        main_target_emb,
        recommended_papers,
        targets,
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
    return gpt_sequence(
        await progress.gather(tasks, desc="Generating related paper summaries")
    )


async def generate_summary_single(
    paper_annotated: gpt.PeerReadAnnotated,
    client: LLMClient,
    user_prompt: PromptTemplate,
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


def single_paper(
    title: Annotated[str, typer.Argument(help="Title of the paper to process")],
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
) -> None:
    """Process a paper title through the complete PETER pipeline and print results."""
    arun_safe(
        process_paper,
        title,
        top_k_refs,
        num_recommendations,
        num_related,
        llm_model,
        encoder_model,
        seed,
        detail,
    )


# TODO: Add Process executor parameter so encoding doesn't block the async loop
async def process_paper_from_title(
    title: str,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
    limiter: Limiter,
) -> gpt.PaperWithRelatedSummary:
    """Process a paper by title through the complete PETER pipeline.

    This function provides a complete end-to-end paper processing pipeline that:
    1. Retrieves paper from Semantic Scholar and arXiv using the title
    2. Processes the paper through the complete PETER pipeline including:
       - S2 reference enhancement with top-k semantic similarity filtering
       - GPT-based annotation extraction (key terms, background, target)
       - Citation context classification (positive/negative polarity)
       - Related paper discovery via citations and semantic matching
       - GPT-generated summaries of related papers

    Args:
        title: Paper title to search for and process.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        llm_model: GPT/Gemini model to use for all LLM API calls.
        encoder_model: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        limiter: Rate limiter for Semantic Scholar API requests to prevent 429 errors.

    Returns:
        Complete paper with related papers and their summaries.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv, or if no
            recommended papers or valid references are found during processing.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    paper = await get_paper_from_title(title, limiter, api_key)

    return await process_paper_complete(
        paper,
        limiter,
        api_key,
        top_k_refs=top_k_refs,
        num_recommendations=num_recommendations,
        num_related=num_related,
        llm_model=llm_model,
        encoder_model=encoder_model,
        seed=seed,
    )


def display_paper_results(result: gpt.PaperWithRelatedSummary) -> None:
    """Display comprehensive results from processed paper.

    Prints a formatted summary of the paper processing results including:
    - Main paper details (title, abstract, key terms, background, target)
    - Citation-based related papers (positive and negative)
    - Semantic-based related papers (positive and negative)
    - Summaries and scores for each related paper

    Args:
        result: Complete processed paper with related papers and summaries.
    """
    print("\n" + "=" * 80)
    print("📊 COMPLETE PAPER PROCESSING RESULTS")
    print("=" * 80)

    # Print main paper information
    print(f"📑 Title: {result.paper.paper.title}")
    print(f"📝 Abstract: {result.paper.paper.abstract}")
    print(
        f"🏷️  Key Terms: {', '.join(list(result.paper.terms.methods) + list(result.paper.terms.tasks))}"
    )
    print(f"🎯 Background: {result.paper.background}")
    print(f"🚀 Target: {result.paper.target}")
    print(f"📊 S2 References: {len(result.paper.paper.references)}")
    print()

    # Print related papers summary
    print(f"🔗 RELATED PAPERS ({len(result.related)} total):")
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
    title: str,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
    detail: bool,
) -> None:
    """Process a paper by title and display results.

    Convenience function that combines processing and display:
    1. Retrieves and processes paper through complete PETER pipeline.
    2. Displays comprehensive results to stdout.

    Args:
        title: Paper title to search for and process.
        top_k_refs: Number of top references to process by semantic similarity.
        num_recommendations: Number of recommended papers to fetch from S2 API.
        num_related: Number of related papers to return for each type
            (citations/semantic, positive/negative).
        llm_model: GPT/Gemini model to use for all LLM API calls.
        encoder_model: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        detail: Show detailed paper information.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv, or if no
            recommended papers or valid references are found during processing.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    limiter = get_limiter(1, 1)  # 1 request per second
    print(f"🔍 Retrieving paper: {title}")

    result = await process_paper_from_title(
        title,
        top_k_refs,
        num_recommendations,
        num_related,
        llm_model,
        encoder_model,
        seed,
        limiter,
    )

    print(f"✅ Found paper: {result.paper.paper.title}")
    print(f"📄 Abstract: {result.paper.paper.abstract[:200]}...")
    print(f"📚 References: {len(result.paper.paper.references)}")
    print(f"📖 Sections: {len(result.paper.paper.sections)}")
    print()
    print("🚀 Processing through PETER pipeline...")

    if detail:
        display_paper_results(result)


def filter_related(
    result: gpt.PaperWithRelatedSummary, pol: rp.ContextPolarity, src: rp.PaperSource
) -> list[PaperRelatedSummarised]:
    """Filter related papers by polarity and source."""
    return [
        r
        for r in result.related
        if r.source.value == src.value and r.polarity.value == pol.value
    ]


def display_related_paper(related: PaperRelatedSummarised) -> str:
    """Display summary of related paper."""
    out = [
        f"    • {related.title}",
        f"      Score: {related.score:.3f}",
        f"      Summary: {related.summary[:100]}...",
    ]
    return "\n".join(out) + "\n"
