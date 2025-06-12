"""Single paper processing pipeline for ORC dataset.

This module provides functionality to process a single paper through the complete PETER
pipeline, including S2 reference enhancement, GPT annotations, context classification,
related paper discovery, and summarization.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

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
from paper.gpt.run_gpt import LLMClient
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
from paper.semantic_scholar.info import fetch_arxiv_papers, get_top_k_titles
from paper.semantic_scholar.model import Paper as S2Paper
from paper.semantic_scholar.model import PaperWithS2Refs
from paper.semantic_scholar.recommended import (
    REQUEST_TIMEOUT,
    fetch_paper_recommendations,
    get_limiter,
)
from paper.util import arun_safe, ensure_envvar, progress
from paper.util.cli import die

if TYPE_CHECKING:
    from paper.gpt.model import PaperRelatedSummarised
    from paper.gpt.prompts import PromptTemplate

logger = logging.getLogger(__name__)


S2_FIELDS = [
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
    "tldr",
    "venue",
]


async def get_paper_from_title(title: str) -> Paper:
    """Get a single processed Paper from a title using Semantic Scholar and arXiv.

    Args:
        title: Paper title to search for.

    Returns:
        Paper object with S2 metadata and parsed arXiv sections/references.

    Raises:
        ValueError: If paper is not found on Semantic Scholar or arXiv.
        RuntimeError: If LaTeX parsing fails or other processing errors occur.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    # Fields to retrieve from S2 API

    # Fetch from Semantic Scholar API
    s2_results = await fetch_arxiv_papers(
        api_key, [title], S2_FIELDS, desc="Fetching paper from S2"
    )

    if not s2_results or not s2_results[0]:
        raise ValueError(f"Paper not found on Semantic Scholar: {title}")

    s2_paper = s2_results[0]

    # Query arXiv for LaTeX content
    openreview_to_arxiv = get_arxiv([s2_paper.title], batch_size=1)

    normalized_title = normalise_title(s2_paper.title)
    arxiv_result = openreview_to_arxiv.get(normalized_title)

    if not arxiv_result:
        raise ValueError(f"Paper not found on arXiv: {s2_paper.title}")

    # Parse arXiv LaTeX
    splitter = SentenceSplitter()
    sections, references = parse_arxiv_latex(arxiv_result, splitter)

    # Create and return Paper object
    return Paper.from_s2(
        s2_paper,
        sections=sections,
        references=references,
    )


async def process_paper_complete(
    paper: Paper,
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

    Returns:
        Complete paper with related papers and their summaries.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """
    # Initialize shared resources
    s2_api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    client = LLMClient.new_env(llm_model, seed)
    encoder = emb.Encoder(encoder_model)

    logger.info("Processing paper: %s", paper.title)

    # Step 3: Fetch S2 recommended papers
    logger.debug("Fetching S2 recommended papers")
    recommended_papers = await fetch_s2_recommendations(
        paper, num_recommendations, s2_api_key
    )
    if not recommended_papers:
        die("no recommended found")

    # Step 1: Enhance with S2 reference data
    logger.debug("Fetching S2 data for top-k references")
    paper_with_s2_refs = await enhance_with_s2_references(
        paper, top_k_refs, encoder, s2_api_key
    )

    # Step 2: Extract key terms and background/target from main paper via GPT
    logger.debug("Extracting key terms and background/target of MAIN PAPER via GPT")
    paper_annotated = await extract_paper_annotations(
        paper_with_s2_refs, client, term_prompt_key, abstract_prompt_key
    )

    # Step 4: Extract background/target from recommended papers via GPT
    logger.debug("Extracting key terms and background/target of RELATED PAPER via GPT")
    recommended_annotated = await extract_recommended_annotations(
        recommended_papers, client, term_prompt_key, abstract_prompt_key
    )

    # Step 5: Classify citation contexts via GPT
    logger.debug("Classifying citation contexts via GPT")
    paper_with_classified_contexts = await classify_citation_contexts(
        paper_with_s2_refs, client, context_prompt_key
    )

    # Step 6: Get related papers (simplified approach without full PETER graphs)
    logger.debug("Getting related papers using direct approach")
    related_papers = await get_related_papers_direct(
        paper_annotated,
        paper_with_classified_contexts,
        recommended_annotated,
        num_related,
        encoder,
    )

    # Step 7: Generate summaries for related papers
    logger.debug("Generating summaries for related papers")
    related_papers_summarised = await generate_related_paper_summaries(
        paper_annotated,
        related_papers,
        client,
        positive_prompt_key,
        negative_prompt_key,
    )

    # Step 8: Create final result
    logger.debug("Creating final result")
    result = gpt.PaperWithRelatedSummary(
        paper=paper_annotated, related=related_papers_summarised
    )

    logger.info("Completed processing paper: %s", paper.title)
    return result


async def enhance_with_s2_references(
    paper: Paper, top_k: int, encoder: emb.Encoder, api_key: str
) -> PaperWithS2Refs:
    """Enhance paper with S2 reference information for top-k similar references."""
    # Get top-k reference titles by semantic similarity
    top_ref_titles = get_top_k_titles(encoder, paper, top_k)

    if not top_ref_titles:
        logger.warning("No references found for paper: %s", paper.title)
        # Return paper with empty S2 references
        return PaperWithS2Refs.from_peer(paper, [])

    # Fetch S2 data for the top references
    fields = [
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
        "tldr",
    ]

    s2_results = await fetch_arxiv_papers(
        api_key, top_ref_titles, fields, desc="Fetching S2 data for references"
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


async def extract_paper_annotations(
    paper_with_s2_refs: PaperWithS2Refs,
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> gpt.PeerReadAnnotated:
    """Extract key terms and background/target information from paper using GPT."""
    # Use specified prompts
    term_prompt = TERM_USER_PROMPTS[term_prompt_key]
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    # Prepare prompts
    term_prompt_text = term_prompt.template.format(
        title=paper_with_s2_refs.title, abstract=paper_with_s2_refs.abstract
    )
    abstract_prompt_text = abstract_prompt.template.format(
        demonstrations="",  # No demonstrations for now
        abstract=paper_with_s2_refs.abstract,
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
        logger.warning("Paper '%s': invalid PaperTerms", paper_with_s2_refs.title)
    if not abstract_classification.is_valid():
        logger.warning(
            "Paper '%s': invalid GPTAbstractClassify", paper_with_s2_refs.title
        )

    # Create PeerReadAnnotated
    return gpt.PeerReadAnnotated(
        terms=terms,
        paper=paper_with_s2_refs,
        background=abstract_classification.background,
        target=abstract_classification.target,
    )


async def extract_recommended_annotations(
    recommended_papers: list[S2Paper],
    client: LLMClient,
    term_prompt_key: str,
    abstract_prompt_key: str,
) -> list[gpt.PaperAnnotated]:
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
    annotated_papers = list(
        await progress.gather(tasks, desc="Annotating related papers.")
    )

    logger.info("Annotated %d recommended papers", len(annotated_papers))
    return annotated_papers


async def extract_recommended_annotations_single(
    client: LLMClient,
    term_prompt: PromptTemplate,
    abstract_prompt: PromptTemplate,
    s2_paper: S2Paper,
) -> gpt.PaperAnnotated:
    """Extract terms and split abstract from paper."""
    # Prepare prompts
    term_prompt_text = term_prompt.template.format(
        title=s2_paper.title, abstract=s2_paper.abstract
    )
    abstract_prompt_text = abstract_prompt.template.format(
        demonstrations="",  # No demonstrations for now
        abstract=s2_paper.abstract,
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
        logger.warning("Recommended paper '%s': invalid PaperTerms", s2_paper.title)
    if not abstract_classification.is_valid():
        logger.warning(
            "Recommended paper '%s': invalid GPTAbstractClassify", s2_paper.title
        )

    # Create annotated paper
    return gpt.PaperAnnotated(
        paper=s2_paper,  # S2Paper is valid as PaperToAnnotate
        terms=terms,
        background=abstract_classification.background,
        target=abstract_classification.target,
    )


async def fetch_s2_recommendations(
    paper: Paper, num_recommendations: int, api_key: str
) -> list[S2Paper]:
    """Fetch recommended papers from S2 API for the given paper."""

    # First, we need to get the S2 paper ID by searching for this paper
    s2_results = await fetch_arxiv_papers(
        api_key,
        [paper.title],
        S2_FIELDS,
        desc="Getting paper ID for recommendations",
    )

    if not s2_results or not s2_results[0]:
        logger.warning(
            "Paper '%s' not found on S2 - cannot fetch recommendations", paper.title
        )
        return []
    s2_paper = s2_results[0]

    # Fields to retrieve for recommended papers
    fields = [
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

    limiter = get_limiter(1, 1)  # 1 request per second

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT), headers={"x-api-key": api_key}
    ) as session:
        return await fetch_paper_recommendations(
            session,
            s2_paper.title,
            s2_paper.paper_id,
            fields,
            num_recommendations,
            limiter,
            from_="recent",
        )


async def classify_citation_contexts(
    paper_with_s2_refs: PaperWithS2Refs,
    client: LLMClient,
    context_prompt_key: str,
) -> gpt.PaperWithContextClassfied:
    """Classify citation contexts by polarity (positive/negative) using GPT."""
    # Load prompts for context classification
    context_prompt = CONTEXT_USER_PROMPTS[context_prompt_key]

    classified_references: list[S2ReferenceClassified] = []
    total_cost = 0

    # Process each reference with S2 data
    # XXX: See if this duplicated
    for reference in paper_with_s2_refs.references:
        classified_contexts: list[ContextClassified] = []

        # Classify each citation context
        for context in reference.contexts:
            user_prompt_text = context_prompt.template.format(
                main_title=paper_with_s2_refs.title,
                main_abstract=paper_with_s2_refs.abstract,
                reference_title=reference.title,
                reference_abstract=reference.abstract,
                context=context.sentence,
            )

            result = await client.run(
                GPTContext, CONTEXT_SYSTEM_PROMPT, user_prompt_text
            )
            total_cost += result.cost

            if gpt_context := result.result:
                classified_contexts.append(
                    ContextClassified(
                        text=context.sentence,
                        gold=context.polarity,  # Original polarity if any
                        prediction=gpt_context.polarity,  # GPT prediction
                    )
                )
            else:
                # Fallback to positive if GPT fails
                classified_contexts.append(
                    ContextClassified(
                        text=context.sentence,
                        gold=context.polarity,
                        prediction=pr.ContextPolarity.POSITIVE,
                    )
                )

        # Create classified reference
        classified_references.append(
            S2ReferenceClassified.from_(reference, contexts=classified_contexts)
        )

    logger.debug("Citation context classification cost: $%.4f", total_cost)

    # Create paper with classified contexts
    return PaperWithContextClassfied(
        title=paper_with_s2_refs.title,
        abstract=paper_with_s2_refs.abstract,
        reviews=paper_with_s2_refs.reviews,
        authors=paper_with_s2_refs.authors,
        sections=paper_with_s2_refs.sections,
        references=classified_references,
        rating=paper_with_s2_refs.rating,
        rationale=paper_with_s2_refs.rationale,
        year=paper_with_s2_refs.year,
    )


async def get_related_papers_direct(
    paper_annotated: gpt.PeerReadAnnotated,
    paper_with_contexts: gpt.PaperWithContextClassfied,
    recommended_papers: list[gpt.PaperAnnotated],
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
) -> list[gpt.PaperRelatedSummarised]:
    """Generate GPT summaries for related papers."""
    if not related_papers:
        return []

    prompt_pol = {
        rp.ContextPolarity.POSITIVE: PETER_SUMMARISE_USER_PROMPTS[positive_prompt_key],
        rp.ContextPolarity.NEGATIVE: PETER_SUMMARISE_USER_PROMPTS[negative_prompt_key],
    }

    summarised_papers: list[gpt.PaperRelatedSummarised] = []
    total_cost = 0

    for related_paper in related_papers:
        # Get the appropriate prompt based on polarity
        user_prompt = prompt_pol[related_paper.polarity]

        # Format prompt using the same function from summarise_related_peter.py
        user_prompt_text = format_template(
            user_prompt,
            paper_annotated,
            related_paper,
        )

        # Generate summary via GPT
        try:
            result = await client.run(
                GPTRelatedSummary, PETER_SUMMARISE_SYSTEM_PROMPT, user_prompt_text
            )
            total_cost += result.cost

            summary = (
                result.result.summary
                if result.result
                else f"Summary not available for: {related_paper.title}"
            )

        except Exception as e:
            logger.warning(
                "Failed to generate summary for '%s': %s", related_paper.title, e
            )
            summary = f"Error generating summary for: {related_paper.title}"

        summarised_papers.append(
            gpt.PaperRelatedSummarised.from_related(related_paper, summary)
        )

    logger.debug("Summary generation cost: $%.4f", total_cost)
    return summarised_papers


def process_paper(
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
) -> None:
    """Process a paper title through the complete PETER pipeline and print results."""
    arun_safe(
        _process_paper_async,
        title,
        top_k_refs,
        num_recommendations,
        num_related,
        llm_model,
        encoder_model,
        seed,
    )


async def _process_paper_async(
    title: str,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder_model: str,
    seed: int,
) -> None:
    """Async implementation of paper processing."""

    # Step 1: Get paper from title
    print(f"🔍 Retrieving paper: {title}")
    paper = await get_paper_from_title(title)
    print(f"✅ Found paper: {paper.title}")
    print(f"📄 Abstract: {paper.abstract[:200]}...")
    print(f"📚 References: {len(paper.references)}")
    print(f"📖 Sections: {len(paper.sections)}")
    print()

    # Step 2: Process paper through complete pipeline
    print("🚀 Processing through PETER pipeline...")
    result = await process_paper_complete(
        paper,
        top_k_refs=top_k_refs,
        num_recommendations=num_recommendations,
        num_related=num_related,
        llm_model=llm_model,
        encoder_model=encoder_model,
        seed=seed,
    )

    # Step 3: Print full results
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
