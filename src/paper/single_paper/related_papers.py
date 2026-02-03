"""Related paper retrieval and summarisation using embeddings and GPT.

This module provides functionality to discover and summarise papers related to a target
paper through two main mechanisms:

1. Citation-based Relationships:
- Identifies papers cited by the target paper.
- Distinguishes between positive (supportive) and negative (contrasting) citations.
- Uses citation context classification to determine relationship polarity.

2. Semantic Similarity:
- Finds papers with similar background concepts or target contributions.
- Uses embedding-based similarity search on paper abstracts.
- Separates background-related papers from target-related papers.

The module operates without building full PETER graphs, instead using a direct approach
that queries only within the paper's citations and recommended papers. It then generates
contextual summaries for the discovered related papers using GPT.

Key Features:
- Embedding-based similarity computation for semantic relationships.
- Polarity-aware citation analysis.
- Parallel GPT summarisation for efficiency.
- Support for different summary prompts based on relationship type.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import related_papers as rp
from paper import semantic_scholar as s2
from paper.gpt.prompts.summarise_related_peter import PETER_SUMMARISE_USER_PROMPTS
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_sequence
from paper.gpt.summarise_related_peter import (
    PETER_SUMMARISE_SYSTEM_PROMPT,
    GPTRelatedSummary,
    format_template,
)
from paper.single_paper.summary_cleaning import clean_summary

if TYPE_CHECKING:
    from paper.gpt.classify_contexts import S2ReferenceClassified
    from paper.gpt.openai_encoder import OpenAIEncoder
    from paper.types import Vector

logger = logging.getLogger(__name__)


def is_empty_string(s: str | None) -> bool:
    """Check if the string is None or empty after stripping whitespace.

    Args:
        s: String to validate.

    Returns:
        True if the string is empty (None or empty after stripping).
    """
    return s is None or not s.strip()


@dataclass(frozen=True, kw_only=True)
class PaperMetadata:
    """Metadata extracted from a paper."""

    year: int | None
    authors: Sequence[str] | None
    venue: str | None
    citation_count: int | None
    reference_count: int | None
    influential_citation_count: int | None
    corpus_id: int | None
    url: str | None
    arxiv_id: str | None


def _extract_paper_metadata(paper: s2.Paper | s2.PaperWithS2Refs) -> PaperMetadata:
    """Extract metadata fields from a paper, handling different paper types."""
    match paper:
        case s2.PaperWithS2Refs():
            return PaperMetadata(
                year=paper.year,
                authors=paper.authors,
                venue=paper.conference,
                arxiv_id=paper.arxiv_id,
                # Not available in PeerRead papers
                citation_count=None,
                reference_count=None,
                influential_citation_count=None,
                corpus_id=None,
                url=None,
            )
        case s2.Paper():
            return PaperMetadata(
                year=paper.year,
                authors=[author.name for author in paper.authors if author.name]
                if paper.authors
                else None,
                venue=paper.venue,
                citation_count=paper.citation_count,
                reference_count=paper.reference_count,
                influential_citation_count=paper.influential_citation_count,
                corpus_id=paper.corpus_id,
                url=paper.url,
                arxiv_id=None,  # Not available in s2.Paper
            )


async def get_related_papers(
    paper_annotated: gpt.PeerReadAnnotated,
    paper_with_contexts: gpt.PaperWithContextClassfied,
    recommended_papers: Sequence[gpt.PaperAnnotated],
    num_related: int,
    encoder: OpenAIEncoder,
    *,
    filter_by_date: bool = False,
) -> list[rp.PaperRelated]:
    """Retrieve related papers through citations and semantic similarity.

    This function implements a direct approach to finding related papers without
    constructing full PETER graphs. It discovers relationships through two channels:

    Citation-based discovery:
    - Extracts papers cited with positive sentiment (supportive references).
    - Extracts papers cited with negative sentiment (contrasting references).
    - Ranks citations by title similarity to the main paper.
    - Returns top K papers for each polarity type.

    Semantic similarity discovery:
    - Finds papers with similar background concepts (negative polarity).
    - Finds papers with similar target contributions (positive polarity).
    - Uses embedding similarity on extracted background/target text.
    - Returns top K papers for each similarity type.

    Args:
        paper_annotated: Main paper with extracted terms and classifications.
        paper_with_contexts: Paper with classified citation contexts.
        recommended_papers: Pool of papers to search for semantic similarity.
        num_related: Number of papers to retrieve per category (this is the K value).
        encoder: Embedding encoder for similarity computations.
        filter_by_date: If True, filter recommended papers to only include those
            published before the main paper. Only affects semantic similarity search.

    Returns:
        Combined list of related papers from all discovery methods, including
        citation-based and semantic similarity-based papers
    """
    logger.debug("Encoding main paper")
    main_title_emb, main_background_emb, main_target_emb = await asyncio.gather(
        encoder.encode(paper_annotated.title),
        encoder.encode(paper_annotated.background),
        encoder.encode(paper_annotated.target),
    )

    # Valid references with non-empty abstracts
    references = [
        r for r in paper_with_contexts.references if not is_empty_string(r.abstract)
    ]
    logger.debug("Getting top K references - positive")
    references_positive_related = await get_top_k_reference_by_polarity(
        encoder, main_title_emb, references, num_related, pr.ContextPolarity.POSITIVE
    )
    logger.debug("Getting top K references - negative")
    references_negative_related = await get_top_k_reference_by_polarity(
        encoder, main_title_emb, references, num_related, pr.ContextPolarity.NEGATIVE
    )

    # Valid recommended papers with non-empty abstracts
    filtered_recommended_papers = [
        r for r in recommended_papers if not is_empty_string(r.abstract)
    ]
    logger.debug(
        "Abstract filtering: %d -> %d papers",
        len(recommended_papers),
        len(filtered_recommended_papers),
    )
    # Filter recommended papers by publication date if requested
    # Use <= to include papers from the same year (avoids filtering all recent work)
    if filter_by_date and paper_annotated.paper.year:
        logger.debug("Filtering by publication date")
        main_year = paper_annotated.paper.year
        filtered_by_date_recommended_papers = [
            paper
            for paper in filtered_recommended_papers
            if paper.paper.year and paper.paper.year <= main_year
        ]
        logger.debug(
            "Date filtering: %d -> %d papers",
            len(filtered_recommended_papers),
            len(filtered_by_date_recommended_papers),
        )
        filtered_recommended_papers = filtered_by_date_recommended_papers
    else:
        if not filter_by_date:
            logger.debug("Date filtering disabled by user")
        if not paper_annotated.paper.year:
            logger.debug("Date filtering unavailable because of missing year")

    logger.debug("Getting top K semantic - background")
    background_related = await get_top_k_semantic(
        encoder,
        num_related,
        main_background_emb,
        filtered_recommended_papers,
        [r.background for r in filtered_recommended_papers],
        pr.ContextPolarity.NEGATIVE,
    )

    logger.debug("Getting top K semantic - target")
    target_related = await get_top_k_semantic(
        encoder,
        num_related,
        main_target_emb,
        filtered_recommended_papers,
        [r.target for r in filtered_recommended_papers],
        pr.ContextPolarity.POSITIVE,
    )

    logger.debug("Positive references: %d", len(references_positive_related))
    logger.debug("Negative references: %d", len(references_negative_related))
    logger.debug("Background related: %d", len(background_related))
    logger.debug("Target related: %d", len(target_related))

    all_related = (
        references_positive_related
        + references_negative_related
        + background_related
        + target_related
    )

    unique_related = deduplicated(
        citations_positive=references_positive_related,
        citations_negative=references_negative_related,
        semantic_positive=target_related,
        semantic_negative=background_related,
    )
    logger.debug("Total related papers with valid abstracts: %d", len(all_related))
    logger.debug("Unique related papers with valid abstracts: %d", len(unique_related))

    return unique_related


def deduplicated(
    citations_positive: Sequence[rp.PaperRelated],
    citations_negative: Sequence[rp.PaperRelated],
    semantic_positive: Sequence[rp.PaperRelated],
    semantic_negative: Sequence[rp.PaperRelated],
) -> list[rp.PaperRelated]:
    """Create a deduplicated version of related papers results using priority rules.

    See also: `related_papers.QueryResult.deduplicated`.

    Returns:
        List of related papers with each paper appearing only once.
    """
    query_result = rp.QueryResult(
        citations_positive=citations_positive,
        citations_negative=citations_negative,
        semantic_positive=semantic_positive,
        semantic_negative=semantic_negative,
    )
    return list(query_result.deduplicated().related)


async def get_top_k_semantic(
    encoder: OpenAIEncoder,
    k: int,
    main_emb: Vector,
    papers: Sequence[gpt.PaperAnnotated],
    items: Sequence[str],
    polarity: pr.ContextPolarity,
) -> list[rp.PaperRelated]:
    """Find top K semantically similar papers based on text embeddings.

    Computes embedding similarity between the main paper's text (background or target)
    and corresponding text from candidate papers. Returns the K most similar papers with
    their similarity scores.

    Args:
        encoder: Embedding encoder for vectorisation.
        k: Number of top papers to return.
        main_emb: Pre-computed embedding of main paper's text.
        papers: Candidate papers to search through.
        items: Text items from papers to embed (backgrounds or targets).
        polarity: Relationship polarity (positive for targets, negative for backgrounds).

    Returns:
        List of K most similar papers as PaperRelated objects with similarity scores.
    """
    if not papers:
        logger.debug("No papers available for semantic search.")
        return []

    # Filter out papers with empty items (OpenAI embeddings API rejects empty strings)
    valid_pairs = [
        (paper, item)
        for paper, item in zip(papers, items, strict=True)
        if not is_empty_string(item)
    ]
    if not valid_pairs:
        logger.debug("No papers with valid items for semantic search.")
        return []

    papers_filtered, items_filtered = zip(*valid_pairs, strict=True)
    sem_emb = await encoder.encode_multi(items_filtered)
    sims = emb.similarities(main_emb, sem_emb)
    top_k = [(papers_filtered[i], float(sims[i])) for i in emb.top_k_indices(sims, k)]

    related_papers: list[rp.PaperRelated] = []
    for paper, score in top_k:
        metadata = _extract_paper_metadata(paper.paper)
        related_papers.append(
            rp.PaperRelated(
                source=rp.PaperSource.SEMANTIC,
                polarity=polarity,
                paper_id=paper.id,
                title=paper.title,
                abstract=paper.abstract,
                score=score,
                year=metadata.year,
                authors=metadata.authors,
                venue=metadata.venue,
                citation_count=metadata.citation_count,
                reference_count=metadata.reference_count,
                influential_citation_count=metadata.influential_citation_count,
                corpus_id=metadata.corpus_id,
                url=metadata.url,
                arxiv_id=metadata.arxiv_id,
                background=paper.background,
                target=paper.target,
            )
        )
    return related_papers


async def get_top_k_reference_by_polarity(
    encoder: OpenAIEncoder,
    title_emb: Vector,
    references: Sequence[S2ReferenceClassified],
    k: int,
    polarity: pr.ContextPolarity,
) -> list[rp.PaperRelated]:
    """Extract top K cited papers filtered by citation polarity.

    Filters references by the specified polarity (positive or negative) and ranks them
    by title similarity to the main paper. This helps identify the most relevant
    citations of each sentiment type.

    Args:
        encoder: Embedding encoder for title vectorisation.
        title_emb: Pre-computed embedding of main paper's title.
        references: All classified references from the paper.
        k: Number of top references to return.
        polarity: Citation polarity to filter by (positive or negative).

    Returns:
        List of K most similar references with the specified polarity, including their
        citation contexts and similarity scores
    """
    references_pol = [r for r in references if r.polarity == polarity]
    if not references_pol:
        logger.debug("No papers with polarity %s found.", polarity)
        return []

    titles_emb = await encoder.encode_multi([r.title for r in references_pol])
    sims = emb.similarities(title_emb, titles_emb)
    top_k = [(references_pol[i], float(sims[i])) for i in emb.top_k_indices(sims, k)]

    return [
        rp.PaperRelated(
            source=rp.PaperSource.CITATIONS,
            polarity=polarity,
            paper_id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            score=score,
            year=paper.year,
            authors=[author.name for author in paper.authors if author.name]
            if paper.authors
            else None,
            venue=paper.venue,
            citation_count=paper.citation_count,
            reference_count=paper.reference_count,
            influential_citation_count=paper.influential_citation_count,
            corpus_id=paper.corpus_id,
            url=paper.url,
            arxiv_id=None,  # Not available in S2ReferenceClassified
            contexts=[
                pr.CitationContext.new(sentence=ctx.text, polarity=ctx.gold)
                for ctx in paper.contexts
            ],
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
    """Generate contextual summaries for all related papers using GPT.

    Creates summaries that explain how each related paper connects to the main paper,
    with different prompts for positive and negative relationships. Processes all papers
    concurrently.

    Args:
        paper_annotated: Main paper with annotations for context.
        related_papers: Papers to summarise.
        client: LLM client for GPT calls.
        positive_prompt_key: Prompt key for positive relationships.
        negative_prompt_key: Prompt key for negative relationships.

    Returns:
        GPTResult containing summarised papers with total API cost.
    """
    if not related_papers:
        return GPTResult(result=[], cost=0)

    prompt_pol = {
        pr.ContextPolarity.POSITIVE: PETER_SUMMARISE_USER_PROMPTS[positive_prompt_key],
        pr.ContextPolarity.NEGATIVE: PETER_SUMMARISE_USER_PROMPTS[negative_prompt_key],
    }

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
    """Generate a contextual summary for one related paper.

    Creates a summary explaining the relationship between the related paper and the main
    paper, using polarity-specific prompts for appropriate context.

    Args:
        paper_annotated: Main paper providing context for the summary.
        client: LLM client for GPT calls.
        user_prompt: Template for generating the summary prompt.
        related_paper: The paper to summarise.

    Returns:
        GPTResult with PaperRelatedSummarised containing the summary and API cost, with
        error fallback to "<error>" message.
    """
    result = await client.run(
        GPTRelatedSummary,
        PETER_SUMMARISE_SYSTEM_PROMPT,
        format_template(
            user_prompt,
            paper_annotated.title,
            paper_annotated.abstract,
            related_paper.title,
            related_paper.abstract,
        ),
    )
    return result.map(
        lambda r: gpt.PaperRelatedSummarised.from_related(
            related_paper, clean_summary(r.summary) if r else "<error>"
        )
    )
