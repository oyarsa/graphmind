"""Related paper retrieval and summarisation.

This module handles finding related papers through citation and semantic similarity,
and generating summaries for the discovered papers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from paper import embedding as emb
from paper import gpt
from paper import peerread as pr
from paper import related_papers as rp
from paper.gpt.run_gpt import GPTResult, LLMClient, gpt_sequence
from paper.gpt.summarise_related_peter import (
    PETER_SUMMARISE_SYSTEM_PROMPT,
    PETER_SUMMARISE_USER_PROMPTS,
    GPTRelatedSummary,
    format_template,
)

if TYPE_CHECKING:
    from paper.gpt.classify_contexts import S2ReferenceClassified

logger = logging.getLogger(__name__)


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
        encoder, main_title_emb, references, num_related, pr.ContextPolarity.POSITIVE
    )
    references_negative_related = get_top_k_reference_by_polarity(
        encoder, main_title_emb, references, num_related, pr.ContextPolarity.NEGATIVE
    )

    background_related = get_top_k_semantic(
        encoder,
        num_related,
        main_background_emb,
        recommended_papers,
        [r.background for r in recommended_papers],
        pr.ContextPolarity.NEGATIVE,
        paper_annotated.background,
        paper_annotated.target,
    )
    target_related = get_top_k_semantic(
        encoder,
        num_related,
        main_target_emb,
        recommended_papers,
        [r.target for r in recommended_papers],
        pr.ContextPolarity.POSITIVE,
        paper_annotated.background,
        paper_annotated.target,
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
    polarity: pr.ContextPolarity,
    background: str | None,
    target: str | None,
) -> list[rp.PaperRelated]:
    """Get top K most similar papers by `items`."""
    sem_emb = encoder.encode_multi(items)
    sims = emb.similarities(main_emb, sem_emb)
    top_k = [(papers[i], float(sims[i])) for i in emb.top_k_indices(sims, k)]

    return [
        rp.PaperRelated(
            source=rp.PaperSource.SEMANTIC,
            polarity=polarity,
            paper_id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            score=score,
            background=background,
            target=target,
        )
        for paper, score in top_k
    ]


def get_top_k_reference_by_polarity(
    encoder: emb.Encoder,
    title_emb: emb.Vector,
    references: Sequence[S2ReferenceClassified],
    k: int,
    polarity: pr.ContextPolarity,
) -> list[rp.PaperRelated]:
    """Get top K references by title similarity."""
    references_pol = [r for r in references if r.polarity == polarity]
    if not references_pol:
        return []

    titles_emb = encoder.encode_multi([r.title for r in references_pol])
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
            contexts=[
                pr.CitationContext(sentence=ctx.text, polarity=ctx.gold)
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
    """Generate GPT summaries for related papers."""
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
    """Generate a single summary for a related paper."""
    result = await client.run(
        GPTRelatedSummary,
        PETER_SUMMARISE_SYSTEM_PROMPT,
        format_template(user_prompt, paper_annotated, related_paper),
    )
    return result.map(
        lambda r: gpt.PaperRelatedSummarised.from_related(
            related_paper, r.summary if r else "<error>"
        )
    )
