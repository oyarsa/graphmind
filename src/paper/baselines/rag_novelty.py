"""RAG-Novelty baseline: retrieve-and-prompt novelty scoring.

Reimplements the RAG-Novelty approach from Lin, Peng, and Fang (AISD@NAACL
2025). The original method embeds paper abstracts, retrieves top-K similar
papers from a corpus, and uses them as context for novelty scoring.

In our adaptation we skip the embedding/retrieval step entirely and reuse the
related papers already collected via PETER (citations + Semantic Scholar
neighbours). This isolates the methodological difference: RAG-Novelty presents
related papers as a flat ranked list, whereas GraphMind structures them by
citation polarity and pairs them with a hierarchical paper graph.

Reference:
    Lin, E., Peng, Z., & Fang, Y. (2025). Evaluating and Enhancing Large
    Language Models for Novelty Assessment in Scholarly Publications.
    Proceedings of the 1st Workshop on AI and Scientific Discovery
    (AISD@NAACL 2025), pp. 46-57. Code: github.com/ethannlin/SchNovel
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper.gpt.model import PaperRelatedSummarised

def format_retrieved_papers(
    papers: Sequence[PaperRelatedSummarised],
) -> str:
    """Format retrieved papers as a numbered list with title, year, and abstract.

    Args:
        papers: Related papers to format (already selected and sorted).

    Returns:
        Formatted string for template substitution.
    """
    if not papers:
        return "[No related papers available]"

    lines: list[str] = []
    for i, paper in enumerate(papers, 1):
        year_str = f" ({paper.year})" if paper.year is not None else ""
        lines.append(f"Paper {i}: {paper.title}{year_str}\nAbstract: {paper.abstract}")

    return "\n\n".join(lines)


def format_rag_novelty_context(
    related: Sequence[PaperRelatedSummarised],
) -> tuple[str, str]:
    """Build the full RAG-Novelty context: retrieved papers and average year note.

    Uses all PETER-collected related papers (typically 15-20 per paper), sorted
    by similarity score. The original RAG-Novelty retrieves from a large corpus
    via embedding similarity; since we reuse PETER's already-relevant papers, we
    include all of them rather than applying a top-K cutoff.

    Args:
        related: All related papers collected via PETER.

    Returns:
        Tuple of (formatted retrieved papers, average year context string).
        The year context is an empty string if no year data is available.
    """
    ranked = sorted(related, key=lambda p: p.score, reverse=True)
    papers_text = format_retrieved_papers(ranked)

    if years := [p.year for p in ranked if p.year is not None]:
        avg_year = round(sum(years) / len(years))
        year_context = (
            f"The average publication year of the retrieved papers is {avg_year}. "
            f"Papers retrieving more recent related work may themselves be more "
            f"novel within their field."
        )
    else:
        year_context = ""

    return papers_text, year_context
