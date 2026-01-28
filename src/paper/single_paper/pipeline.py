"""Main pipeline orchestration for single paper processing.

This module contains the high-level pipeline functions that coordinate
the complete PETER pipeline workflow.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from paper import gpt
from paper import peerread as pr
from paper.embedding import DEFAULT_SENTENCE_MODEL as DEFAULT_SENTENCE_MODEL
from paper.gpt.run_gpt import GPTResult, LLMClient
from paper.orc.arxiv_api import arxiv_id_from_url
from paper.single_paper.annotation_extraction import (
    classify_citation_contexts,
    extract_main_paper_annotations,
    extract_recommended_annotations,
)
from paper.single_paper.graph_evaluation import (
    EvaluationResult,
    evaluate_paper_with_graph,
)
from paper.single_paper.paper_retrieval import (
    REQUEST_TIMEOUT,
    ProgressCallback,
    enhance_with_s2_references,
    fetch_s2_recommendations,
    get_paper_from_arxiv_id,
    get_paper_from_title,
)
from paper.single_paper.related_papers import (
    generate_related_paper_summaries,
    get_related_papers,
)
from paper.util import atimer, ensure_envvar
from paper.util.cli import die

if TYPE_CHECKING:
    from paper.gpt.openai_encoder import OpenAIEncoder
    from paper.util.rate_limiter import Limiter

logger = logging.getLogger(__name__)


async def annotate_paper_pipeline(
    paper: pr.Paper,
    limiter: Limiter,
    s2_api_key: str,
    client: LLMClient,
    encoder: OpenAIEncoder,
    *,
    top_k_refs: int = 20,
    num_recommendations: int = 30,
    num_related: int = 2,
    term_prompt_key: str = "multi",
    abstract_prompt_key: str = "simple",
    context_prompt_key: str = "sentence",
    positive_prompt_key: str = "positive",
    negative_prompt_key: str = "negative",
    request_timeout: float = REQUEST_TIMEOUT,
    filter_by_date: bool = False,
    callback: ProgressCallback | None = None,
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
        filter_by_date: If True, filter recommended papers to only include those
            published before the main paper.
        s2_api_key: Semantic Scholar API key.
        client: LLM client for GPT API calls.
        encoder: Embedding encoder for semantic similarity computations.
        callback: Optional callback function to call with phase names after completion.

    Returns:
        Complete paper with related papers and their summaries wrapped in GPTResult.

    Raises:
        ValueError if no recommended papers or valid references were found.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY and OPENAI_API_KEY/GEMINI_API_KEY environment variables.
    """

    logger.debug("Processing paper: %s", paper.title)

    # Phase 1: Fetch S2 recommended papers and reference data in parallel
    if callback:
        await callback("Fetching Semantic Scholar references and recommendations")

    logger.debug("Fetching S2 data for recommendations and references in parallel")
    paper_with_s2_refs, recommended_papers = await asyncio.gather(
        atimer(
            enhance_with_s2_references(paper, top_k_refs, encoder, s2_api_key, limiter),
            3,
        ),
        atimer(
            fetch_s2_recommendations(
                paper, num_recommendations, s2_api_key, limiter, request_timeout
            ),
            3,
        ),
    )

    if not paper_with_s2_refs or not recommended_papers:
        raise ValueError("Could not find any related papers")

    # Phase 2: Extract annotations from all papers and classify contexts in parallel
    if callback:
        await callback("Extracting annotations and classifying contexts")

    logger.debug("Processing all annotations and citation contexts in parallel")
    (
        paper_annotated,
        recommended_annotated,
        paper_with_classified_contexts,
    ) = await asyncio.gather(
        atimer(
            extract_main_paper_annotations(
                paper_with_s2_refs, client, term_prompt_key, abstract_prompt_key
            ),
            3,
        ),
        atimer(
            extract_recommended_annotations(
                recommended_papers, client, term_prompt_key, abstract_prompt_key
            ),
            3,
        ),
        atimer(
            classify_citation_contexts(paper_with_s2_refs, client, context_prompt_key),
            3,
        ),
    )

    # Phase 3: Get related papers (simplified approach without full PETER graphs)
    if callback:
        await callback("Discovering related papers")

    logger.debug("Getting related papers using direct approach")
    related_papers = await atimer(
        get_related_papers(
            paper_annotated.result,
            paper_with_classified_contexts.result,
            recommended_annotated.result,
            num_related,
            encoder,
            filter_by_date=filter_by_date,
        ),
        3,
    )

    # Phase 4: Generate summaries for related papers
    if callback:
        await callback("Generating related paper summaries")

    logger.debug("Generating summaries for related papers")
    related_papers_summarised = await atimer(
        generate_related_paper_summaries(
            paper_annotated.result,
            related_papers,
            client,
            positive_prompt_key,
            negative_prompt_key,
        ),
        3,
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


class QueryType(StrEnum):
    """Whether to query arXiv by title or ID/URL."""

    TITLE = "title"
    ID = "id"


async def process_paper_from_query(
    query: str,
    type_: QueryType,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    llm_model: str,
    encoder: OpenAIEncoder,
    seed: int,
    limiter: Limiter,
    eval_prompt_key: str,
    graph_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
    interactive: bool = False,
) -> EvaluationResult:
    """Process a paper by query (title or arXiv ID/URL) through the PETER pipeline.

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
        encoder: Embedding encoder model for semantic similarity computations.
        seed: Random seed for GPT API calls to ensure reproducibility.
        limiter: Rate limiter for Semantic Scholar API requests to prevent 429 errors.
        eval_prompt_key: Key for evaluation prompt template.
        graph_prompt_key: Key for graph extraction prompt template.
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.
        interactive: If True, enables interactive paper selection for title searches.

    Returns:
        EvaluationResult with novelty evaluation and cost.

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
            paper = await atimer(get_paper_from_title(query, limiter, s2_api_key), 2)
        case QueryType.ID:
            if interactive:
                die("Interactive mode is only supported for title search")

            arxiv_id = arxiv_id_from_url(query)
            logger.debug("Searching by arXiv ID: %s", arxiv_id)
            paper = await atimer(
                get_paper_from_arxiv_id(arxiv_id, limiter, s2_api_key), 2
            )

    # Process paper through PETER pipeline
    paper_result = await atimer(
        annotate_paper_pipeline(
            paper=paper,
            limiter=limiter,
            s2_api_key=s2_api_key,
            client=client,
            encoder=encoder,
            top_k_refs=top_k_refs,
            num_recommendations=num_recommendations,
            num_related=num_related,
        ),
        2,
    )

    # Evaluate with graph
    graph_result = await atimer(
        evaluate_paper_with_graph(
            paper_result.result,
            client,
            eval_prompt_key,
            graph_prompt_key,
            demonstrations_key,
            demo_prompt_key,
        ),
        2,
    )

    return EvaluationResult.from_(paper_result.then(graph_result))


async def process_paper_from_selection(
    client: LLMClient,
    title: str,
    arxiv_id: str,
    top_k_refs: int,
    num_recommendations: int,
    num_related: int,
    limiter: Limiter,
    encoder: OpenAIEncoder,
    eval_prompt_key: str,
    graph_prompt_key: str,
    demonstrations_key: str,
    demo_prompt_key: str,
    *,
    filter_by_date: bool = False,
    callback: ProgressCallback | None = None,
) -> EvaluationResult:
    """Process a paper from pre-selected title and arXiv ID through the PETER pipeline.

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
    4. Extracts a graph representation and evaluates novelty

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
        demonstrations_key: Key for demonstrations file.
        demo_prompt_key: Key for demonstration prompt template.
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
        evaluate_paper_with_graph(
            paper_annotated.result,
            client,
            eval_prompt_key,
            graph_prompt_key,
            demonstrations_key,
            demo_prompt_key,
            callback=callback,
        ),
        2,
    )

    return EvaluationResult.from_(paper_annotated.then(graph_evaluated))
