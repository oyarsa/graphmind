"""Model definitions, costs, and search level configuration for LLM APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum


class SearchLevel(str, Enum):
    """Search context size levels for web search/grounding in LLM APIs.

    Determines how much search data is retrieved and processed.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_str(cls, value: str | SearchLevel | None) -> SearchLevel | None:
        """Convert string to SearchLevel enum, or return None if value is None.

        Args:
            value: String representation of search level, or existing SearchLevel.

        Returns:
            SearchLevel enum value, or None if input is None.

        Raises:
            ValueError: If the string is not a valid search level.
        """
        if value is None:
            return None

        if isinstance(value, SearchLevel):
            return value

        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid search level: {value}. Must be one of: low, medium, high"
            ) from e


MODEL_SYNONYMS: Mapping[str, str] = {
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
    "4o-search": "gpt-4o-search-preview-2025-03-11",
    "4o-mini-search": "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-search": "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search": "gpt-4o-mini-search-preview-2025-03-11",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro": "gemini-2.5-pro-preview-03-25",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.2": "gpt-5.2-2025-12-11",
}
"""Mapping between short and common model names and their full versioned names."""

MODELS_ALLOWED: Sequence[str] = sorted(MODEL_SYNONYMS.keys() | MODEL_SYNONYMS.values())
"""All allowed model names, including synonyms and full names."""

MODEL_COSTS: Mapping[str, tuple[float, float]] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-mini-search-preview-2025-03-11": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
    "gpt-4o-search-preview-2025-03-11": (2.5, 10),
    "gemini-2.0-flash-001": (0.10, 0.40),
    "gemini-2.5-pro-preview-03-25": (1.25, 2.5),
    "gemini-2.5-flash-preview-04-17": (0.15, 0.60),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2, 8),
    "gpt-5-mini-2025-08-07": (0.25, 2),
    "gpt-5-nano-2025-08-07": (0.05, 0.40),
    "gpt-5-2025-08-07": (1.75, 14),
    "gpt-5.2-2025-12-11": (1.75, 14),
}
"""Cost in $ per 1M tokens: (input cost, output cost).

From https://openai.com/api/pricing/
"""

MODEL_SEARCH_COSTS: Mapping[str, Mapping[SearchLevel, float]] = {
    "gpt-4o-mini-search-preview-2025-03-11": {
        SearchLevel.LOW: 0.025,  # $25 per 1000 searches
        SearchLevel.MEDIUM: 0.025,  # Assuming same as low (exact pricing not specified)
        SearchLevel.HIGH: 0.025,  # Assuming same as low (exact pricing not specified)
    },
    "gpt-4o-search-preview-2025-03-11": {
        SearchLevel.LOW: 0.030,  # $30 per 1000 searches
        SearchLevel.MEDIUM: 0.040,  # Estimated midpoint
        SearchLevel.HIGH: 0.050,  # $50 per 1000 searches
    },
    "gemini-2.0-flash-001": {
        SearchLevel.LOW: 0.035,  # $35 per 1000 grounded queries
        SearchLevel.MEDIUM: 0.035,  # Same for all context sizes
        SearchLevel.HIGH: 0.035,  # Same for all context sizes
    },
    "gemini-2.5-pro-preview-03-25": {
        SearchLevel.LOW: 0.035,  # $35 per 1000 grounded queries
        SearchLevel.MEDIUM: 0.035,  # Same for all context sizes
        SearchLevel.HIGH: 0.035,  # Same for all context sizes
    },
    "gemini-2.5-flash-preview-04-17": {
        SearchLevel.LOW: 0.035,  # $35 per 1000 grounded queries
        SearchLevel.MEDIUM: 0.035,  # Same for all context sizes
        SearchLevel.HIGH: 0.035,  # Same for all context sizes
    },
}
"""Cost in $ per 1000 search/grounding calls, by context size level.

From https://openai.com/api/pricing/ and
https://ai.google.dev/gemini-api/docs/google-search
"""

MODELS_ALLOWED_AZURE = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5-mini",
}
"""All allowed model names from the Azure API."""

AZURE_TIER = 10
"""Separate tier for Azure API."""


def calc_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    used_search: bool = False,
    search_level: SearchLevel | None = None,
) -> float:
    """Calculate API request cost based on model, tokens, and search usage.

    NB: prompt_tokens/completion_tokens is the name given to input/output tokens in the
    usage object from the OpenAI result.

    Args:
        model: Model identifier key.
        prompt_tokens: The input tokens for the API.
        completion_tokens: The output tokens from the API.
        used_search: Whether the request actually used search/grounding. For OpenAI,
            this is always True when search_level is provided. For Gemini, this depends
            on whether grounding was actually used (presence of grounding metadata).
        search_level: The search context size level used. Only relevant if `used_search`
            is True.

    Returns:
        The total cost of the request in dollars. If the model is invalid, returns 0.
    """
    if model not in MODEL_COSTS:
        return 0

    # Calculate token costs
    input_cost, output_cost = MODEL_COSTS[model]
    token_cost = (
        prompt_tokens / 1e6 * input_cost + completion_tokens / 1e6 * output_cost
    )

    # Calculate search costs if applicable
    search_cost = 0.0
    if used_search and search_level is not None and model in MODEL_SEARCH_COSTS:
        # Search costs are per 1000 calls, so we divide by 1000
        search_cost = MODEL_SEARCH_COSTS[model][search_level] / 1000

    return token_cost + search_cost


def resolve_model_name(model: str) -> str:
    """Resolve model synonym to full model name."""
    return MODEL_SYNONYMS.get(model, model)
