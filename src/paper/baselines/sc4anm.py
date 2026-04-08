"""SC4ANM baseline: IMRaD section extraction and truncation.

Reimplements the GPT prompting path from SC4ANM (Wu et al., 2025), which
segments papers into IMRaD sections and prompts an LLM with the optimal
section combination (Introduction + Results + Discussion) for novelty
prediction.

The section classification uses heading-based heuristic matching rather than
SC4ANM's SciBERT classifier, since our papers are already parsed from LaTeX
into Markdown with section headings preserved.

Reference:
    Wu, W., Zhang, C., Bao, T., & Zhao, Y. (2025). SC4ANM: Identifying
    Optimal Section Combinations for Automated Novelty Prediction in Academic
    Papers. Expert Systems with Applications.
    Code: github.com/njust-winchy/SC4ANM
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paper.gpt.tokenizer import truncate_text

if TYPE_CHECKING:
    from paper.peerread.model import PaperSection


# --------------------------------------------------------------------------- #
# Heading classification patterns (case-insensitive)
# --------------------------------------------------------------------------- #

_INTRODUCTION = re.compile(r"introduction", re.IGNORECASE)
_METHODS = re.compile(
    r"method|methodology|approach|framework|model|proposed|technique",
    re.IGNORECASE,
)
_RESULTS = re.compile(
    r"result|experiment|evaluation|empirical|analysis",
    re.IGNORECASE,
)
_DISCUSSION = re.compile(
    r"discussion|conclusion|limitation|future.work|summary",
    re.IGNORECASE,
)
_EXCLUDED = re.compile(
    r"related.work|background|preliminary|appendix|acknowledge|reference",
    re.IGNORECASE,
)

_IMRAD_PATTERNS = {
    "introduction": _INTRODUCTION,
    "methods": _METHODS,
    "results": _RESULTS,
    "discussion": _DISCUSSION,
}

# Per-section token budget. SC4ANM §4 uses 2000, but empirical testing showed
# that shorter truncation (500 tokens) reduces the model's tendency to
# over-predict novelty from self-promotional section text.
DEFAULT_MAX_SECTION_TOKENS = 500

_SECTION_NOT_AVAILABLE = "[Section not available]"


@dataclass(frozen=True, kw_only=True)
class IMRaDSections:
    """Classified IMRaD sections extracted from a paper.

    Each field is the concatenated text of all sections whose heading matched
    the corresponding IMRaD category, or ``None`` if no heading matched.
    """

    introduction: str | None
    methods: str | None
    results: str | None
    discussion: str | None


def classify_sections(sections: Sequence[PaperSection]) -> IMRaDSections:
    """Classify paper sections into IMRaD categories by heading matching.

    Headings are matched against fixed regex patterns (see module-level
    constants). Sections whose headings match the exclusion list (related work,
    background, appendix, etc.) are skipped.  If multiple sections match the
    same IMRaD category, their texts are concatenated with a blank line.

    Args:
        sections: Parsed paper sections with ``heading`` and ``text`` fields.

    Returns:
        An ``IMRaDSections`` instance with text for each matched category or
        ``None`` for unmatched categories.
    """
    buckets: dict[str, list[str]] = {cat: [] for cat in _IMRAD_PATTERNS}

    for section in sections:
        heading = section.heading.strip()
        if _EXCLUDED.search(heading):
            continue

        for category, pattern in _IMRAD_PATTERNS.items():
            if pattern.search(heading):
                buckets[category].append(section.text)
                break

    def join(texts: list[str]) -> str | None:
        return "\n\n".join(texts) if texts else None

    return IMRaDSections(
        introduction=join(buckets["introduction"]),
        methods=join(buckets["methods"]),
        results=join(buckets["results"]),
        discussion=join(buckets["discussion"]),
    )


def format_ird_sections(
    sections: IMRaDSections,
    *,
    max_tokens: int = DEFAULT_MAX_SECTION_TOKENS,
) -> str:
    """Format Introduction + Results + Discussion sections for the SC4ANM prompt.

    Each present section is truncated to *max_tokens* tokens (cl100k_base).
    Missing sections are replaced with "[Section not available]".

    Args:
        sections: Classified IMRaD sections.
        max_tokens: Per-section token budget.

    Returns:
        Formatted multi-section string ready for template substitution.
    """
    intro = (
        truncate_text(sections.introduction, max_tokens)
        if sections.introduction
        else _SECTION_NOT_AVAILABLE
    )
    results = (
        truncate_text(sections.results, max_tokens)
        if sections.results
        else _SECTION_NOT_AVAILABLE
    )
    discussion = (
        truncate_text(sections.discussion, max_tokens)
        if sections.discussion
        else _SECTION_NOT_AVAILABLE
    )

    return f"Introduction:\n{intro}\n\nResults:\n{results}\n\nDiscussion:\n{discussion}"
