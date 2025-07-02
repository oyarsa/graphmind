"""Functions to partially evaluate a paper from its title and abstract."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, Field

from paper import gpt
from paper import peerread as pr
from paper.backend.model import DEMO_PROMPT, DEMOS, PartialEvaluationResponse
from paper.gpt.annotate_paper import (
    ABS_SYSTEM_PROMPT,
    ABS_USER_PROMPTS,
    GPTAbstractClassify,
)
from paper.gpt.evaluate_paper import GPTStructured, GPTStructuredRaw, get_demonstrations
from paper.gpt.evaluate_paper_graph import GRAPH_EVAL_USER_PROMPTS, format_related
from paper.gpt.novelty_utils import get_novelty_probability
from paper.gpt.run_gpt import GPTResult, gpt_sequence, gpt_unit
from paper.gpt.summarise_related_peter import (
    PETER_SUMMARISE_SYSTEM_PROMPT,
    PETER_SUMMARISE_USER_PROMPTS,
    GPTRelatedSummary,
    format_template,
)
from paper.single_paper.partial_search import search_related_papers
from paper.single_paper.related_papers import get_top_k_semantic
from paper.util import ensure_envvar

if TYPE_CHECKING:
    from paper import embedding as emb
    from paper import related_papers as rp
    from paper import semantic_scholar as s2
    from paper.gpt.model import PaperRelatedSummarised
    from paper.gpt.prompts import PromptTemplate
    from paper.gpt.run_gpt import LLMClient
    from paper.single_paper.paper_retrieval import ProgressCallback
    from paper.util.rate_limiter import Limiter


logger = logging.getLogger(__name__)


async def partial_evaluation(
    client: LLMClient,
    limiter: Limiter,
    encoder: emb.Encoder,
    num_recommendations: int,
    num_semantic: int,
    title: str,
    abstract: str,
    callback: ProgressCallback,
    demonstrations_key: str = DEMOS,
    demo_prompt_key: str = DEMO_PROMPT,
    abstract_prompt_key: str = "simple",
    positive_prompt_key: str = "positive",
    negative_prompt_key: str = "negative",
) -> PartialEvaluationResponse:
    """Partially evaluate a paper from its title and abstract.

    Retrieves similar papers, finds semantic related papers and generates a rationale
    from this limited data.

    Args:
        client: LLM client to use.
        limiter: Rate limiter to use for Semantic Scholar API requests.
        encoder: Text encoder used to generate embeddings.
        num_recommendations: How many papers to retrieve from the Semantic Scholar API,
            which will be used as input for the related papers step.
        num_semantic: How many semantic papers of each type to retrieve.
        title: Paper title.
        abstract: Paper abstract.
        callback: Callback to call to report progress.
        use_keywords: Whether to extract keywords from the abstract using an LLM to
            augment the recommendation search.
        abstract_prompt_key: Key for abstract classification prompt template.
        positive_prompt_key: Prompt key for positive relationships.
        negative_prompt_key: Prompt key for negative relationships.
        demonstrations_key: Key for demonstration examples.
        demo_prompt_key: Key for demonstration prompt template.

    Returns:
        Partial evaluation result.

    Requires:
        SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    abstract_prompt = ABS_USER_PROMPTS[abstract_prompt_key]

    await callback("Extracting keywords from abstract")
    keywords = await extract_keywords(client, title, abstract)
    logger.debug("Extracted keywords: %s", keywords)

    await callback("Searching related papers")
    related = await search_related_papers(
        limiter,
        api_key,
        title,
        abstract,
        keywords=keywords.result,
        max_results=num_recommendations,
    )
    logger.debug("Recommended retrieved: %d", len(related))

    await callback("Extracting background and target from abstracts")
    recommended_annotated, main_annotated = await asyncio.gather(
        extract_background_target_related(related, client, abstract_prompt),
        extract_background_target(abstract, client, abstract_prompt),
    )
    logger.debug("Recommended annotated: %d", len(recommended_annotated.result))

    if not recommended_annotated.result:
        raise RuntimeError("Could not retrieve relevant papers")

    background = main_annotated.result.background
    target = main_annotated.result.target

    await callback("Retrieving semantic papers")
    semantic = await retrieve_semantic_papers(
        encoder, background, target, recommended_annotated.result, num_semantic
    )

    await callback("Summarising related papers")
    summaries = await generate_related_paper_summaries(
        title, abstract, semantic, client, positive_prompt_key, negative_prompt_key
    )

    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    await callback("Evaluating partial paper")
    evaluation = await evaluate_partial_paper(
        title, abstract, summaries.result, client, demonstrations
    )

    total_cost = (
        keywords.cost
        + recommended_annotated.cost
        + main_annotated.cost
        + summaries.cost
        + evaluation.cost
    )

    return PartialEvaluationResponse(
        title=title,
        abstract=abstract,
        keywords=keywords.result,
        label=evaluation.result.label,
        probability=evaluation.result.probability,
        paper_summary=evaluation.result.paper_summary,
        supporting_evidence=evaluation.result.supporting_evidence,
        contradictory_evidence=evaluation.result.contradictory_evidence,
        conclusion=evaluation.result.conclusion,
        total_cost=total_cost,
        related=summaries.result,
    )


async def extract_background_target(
    abstract: str, client: LLMClient, prompt: gpt.PromptTemplate
) -> GPTResult[GPTAbstractClassify]:
    """Extract target and background from a paper abstract using GPT.

    Args:
        abstract: Paper abstract to analyse.
        client: LLM client for GPT calls.
        prompt: Template for abstract classification prompts.

    Returns:
        GPTResult containing the abstract extraction results, along with total
        API cost.
    """
    abstract_prompt_text = prompt.template.format(
        demonstrations="",
        abstract=abstract,
    )

    result = await client.run(
        GPTAbstractClassify, ABS_SYSTEM_PROMPT, abstract_prompt_text
    )
    return result.map(lambda r: r or GPTAbstractClassify.empty())


async def extract_background_target_related(
    related: Sequence[s2.Paper], client: LLMClient, prompt: gpt.PromptTemplate
) -> GPTResult[Sequence[gpt.PaperAnnotated]]:
    """Extract background and target from a sequence of related papers' abstracts.

    Args:
        related: Sequence of related papers to extract background and target from.
        client: LLM client for GPT calls.
        prompt: Template for abstract classification prompts.

    Returns:
        GPTResult containing a sequence of abstract extraction results for each related
        paper.
    """
    tasks = [
        extract_background_target(p.abstract or "", client, prompt) for p in related
    ]
    return gpt_sequence(await asyncio.gather(*tasks)).map(
        lambda annotated: [
            gpt.PaperAnnotated(
                terms=gpt.PaperTerms.empty(),
                paper=paper,
                background=abs.background,
                target=abs.target,
            )
            for paper, abs in zip(related, annotated)
        ]
    )


async def retrieve_semantic_papers(
    encoder: emb.Encoder,
    background: str,
    target: str,
    related: Sequence[gpt.PaperAnnotated],
    k: int,
) -> list[rp.PaperRelated]:
    """Retrieve semantic related papers based on the background and target.

    Args:
        encoder: Text encoder used to generate embeddings.
        background: Background text to use for semantic search.
        target: Target text to use for semantic search.
        related: Sequence of related papers to search within.
        k: Number of top K semantic related papers to retrieve, for each type. 2*K in
            total.

    Returns:
        List of semantic related papers, combining background and target results.
    """
    main_background_emb = encoder.encode(background)
    main_target_emb = encoder.encode(target)

    logger.debug("Getting top K semantic - background - from %d papers", len(related))
    background_related = get_top_k_semantic(
        encoder,
        k,
        main_background_emb,
        related,
        [r.background for r in related],
        pr.ContextPolarity.NEGATIVE,
    )

    logger.debug("Getting top K semantic - target - from %d papers", len(related))
    target_related = get_top_k_semantic(
        encoder,
        k,
        main_target_emb,
        related,
        [r.target for r in related],
        pr.ContextPolarity.POSITIVE,
    )

    return background_related + target_related


class Keywords(BaseModel):
    """Structured type for extracting keywords from a paper's title and abstract."""

    keywords: Annotated[
        Sequence[str], Field(description="Keywords that represent the paper. Up to 10.")
    ]


KEYWORDS_SYSTEM_PROMPT = """
You are an expert academic keyword extractor. Your task is to analyze paper titles and
abstracts to extract keywords that would be most effective for finding related papers in
academic databases.

Extract keywords following these guidelines:

1. Focus on core concepts, methodologies, techniques, and application domains
2. Include both single-word terms and multi-word phrases (2-4 words)
3. Aim for the "goldilocks zone" of specificity - not too broad, not too narrow
4. Include acronyms if they are standard in the field
5. Extract 5-10 keywords total
6. Use the exact terminology from the text when possible
7. Prioritize terms that distinguish this work from others in its specific area

Keywords should be:
- Specific to the paper's research area and approach
- Terms that similar papers in this niche would likely use
- More specific than broad field names (e.g., not "machine learning" but rather
  "graph neural networks" or "few-shot learning")
- Less specific than unique contributions of this exact paper

Examples of good specificity levels:
- Instead of "computer vision" → "semantic segmentation" or "object detection"
- Instead of "natural language processing" → "named entity recognition" or
  "transformer models"
- Instead of "optimization" → "convex optimization" or "evolutionary algorithms"

Do not include:
- Generic academic terms like "research", "study", "analysis"
- Broad field names like "machine learning", "artificial intelligence", "computer science"
- Author names or institution names
- Terms unique to only this paper's specific contribution
"""
KEYWORDS_USER_TEMPLATE = """
Extract keywords from the following paper that would be most useful for finding related
work on Semantic Scholar:

Title: {title}

Abstract: {abstract}

Extract keywords that capture the main concepts, methods, techniques, and application
areas of this paper.
"""


async def extract_keywords(
    client: LLMClient, title: str, abstract: str
) -> GPTResult[Sequence[str]]:
    """Extract keywords from the paper's title and abstract using an LLM.

    Args:
        client: LLM client to use for keyword extraction.
        title: Paper title.
        abstract: Paper abstract.

    Returns:
        A list of keywords representing the paper.
    """
    user_prompt = KEYWORDS_USER_TEMPLATE.format(title=title, abstract=abstract)

    result = await client.run(
        Keywords, system_prompt=KEYWORDS_SYSTEM_PROMPT, user_prompt=user_prompt
    )
    return result.map(lambda r: r.keywords if r else [])


async def evaluate_partial_paper(
    title: str,
    abstract: str,
    related: Sequence[PaperRelatedSummarised],
    client: LLMClient,
    demonstrations: str,
) -> GPTResult[GPTStructured]:
    """Evaluate a paper's novelty using the extracted graph.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        related: Related papers to use for evaluation.
        client: LLM client for GPT API calls.
        demonstrations: Demonstration examples.

    Returns:
        Evaluation result wrapped in GPTResult.
    """
    eval_prompt = GRAPH_EVAL_USER_PROMPTS["simple-basic"]
    result = await client.run(
        GPTStructuredRaw,
        eval_prompt.system,
        format_eval_template(eval_prompt, title, abstract, related, demonstrations),
        logprobs=True,
        top_logprobs=3,
    )

    eval = result.map(
        lambda r: r.with_prob(get_novelty_probability(result.logprobs))
        if r is not None
        else GPTStructured.error()
    )
    if not eval.result.is_valid():
        logger.warning(f"Paper '{title}': invalid evaluation result")

    return eval.map(
        lambda r: r.with_prob(get_novelty_probability(eval.logprobs))
    ).nologits()


def format_eval_template(
    eval_prompt: PromptTemplate,
    title: str,
    abstract: str,
    related: Sequence[PaperRelatedSummarised],
    demonstrations: str,
) -> str:
    """Format the evaluation prompt template with the partial data.

    Args:
        eval_prompt: Evaluation prompt template to use.
        title: Paper title.
        abstract: Paper abstract.
        related: Summarised related papers to use for evaluation.
        demonstrations: Demonstration examples to include in the prompt.

    Returns:
        Formatted evaluation prompt string.
    """
    return eval_prompt.template.format(
        demonstrations=demonstrations,
        title=title,
        abstract=abstract,
        positive=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.POSITIVE
        ),
        negative=format_related(
            p for p in related if p.polarity is pr.ContextPolarity.NEGATIVE
        ),
    )


async def generate_related_paper_summaries(
    title: str,
    abstract: str,
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
        title: Title of the main paper.
        abstract: Abstract of the main paper.
        related_papers: Papers to summarise.
        client: LLM client for GPT calls.
        positive_prompt_key: Prompt key for positive relationships.
        negative_prompt_key: Prompt key for negative relationships.

    Returns:
        GPTResult containing summarised papers with total API cost.
    """
    if not related_papers:
        return gpt_unit([])

    prompt_pol = {
        pr.ContextPolarity.POSITIVE: PETER_SUMMARISE_USER_PROMPTS[positive_prompt_key],
        pr.ContextPolarity.NEGATIVE: PETER_SUMMARISE_USER_PROMPTS[negative_prompt_key],
    }

    tasks = [
        generate_summary_single(
            title, abstract, client, prompt_pol[related_paper.polarity], related_paper
        )
        for related_paper in related_papers
    ]
    return gpt_sequence(await asyncio.gather(*tasks))


async def generate_summary_single(
    title: str,
    abstract: str,
    client: LLMClient,
    user_prompt: gpt.PromptTemplate,
    related_paper: rp.PaperRelated,
) -> GPTResult[gpt.PaperRelatedSummarised]:
    """Generate a contextual summary for one related paper.

    Creates a summary explaining the relationship between the related paper and the main
    paper, using polarity-specific prompts for appropriate context.

    Args:
        title: Title of the main paper.
        abstract: Abstract of the main paper.
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
            title,
            abstract,
            related_paper.title,
            related_paper.abstract,
        ),
    )
    return result.map(
        lambda r: gpt.PaperRelatedSummarised.from_related(
            related_paper, r.summary if r else "<error>"
        )
    )
