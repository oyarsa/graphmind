"""Rate limiting configuration for LLM APIs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from paper.gpt.models import AZURE_TIER
from paper.util.rate_limiter import ChatRateLimiter

logger = logging.getLogger(__name__)


def get_api_tier(provider: str, base_url: str | None = None) -> int:
    """Get API tier from environment or defaults.

    Args:
        provider: The API provider.
        base_url: Base URL for the API (used to detect Azure).

    Returns:
        The API tier number.
    """
    if provider == "azure":
        return AZURE_TIER

    if provider == "openai" and base_url is not None:
        # Custom OpenAI-compatible endpoint
        return -1

    key_var = f"{provider.upper()}_API_TIER"
    if provider in ("openai", "gemini") and (api_tier_s := os.getenv(key_var)):
        return int(api_tier_s)

    logger.warning(f"{key_var} unset. Defaulting to tier 1.")
    return 1


def get_rate_limiter(tier: int, model: str) -> ChatRateLimiter:
    """Get the rate limiter for a specific model based on the API tier.

    Args:
        tier: Tier the organisation is in.
        model: API model name.

    Returns:
        Rate limiter for the model with the correct rate limits for the tier.

    Raises:
        ValueError if tier or model are invalid.
    """
    message = (
        "Tier {tier} limits are not set. Please provide the limits. You can find them on"
        " https://platform.openai.com/settings/organization/limits, using the"
        " `scripts/tools/rate_limits.py` tool or on https://ai.google.dev/gemini-api/docs/rate-limits#tier-1"
    )

    # <request_limit, token_limit> per minute
    limits: dict[str, tuple[int, int]]

    if tier == -1:
        rate_limits = (1_000, 1_000_000)
    else:
        if tier == 1:
            limits = {
                "gemini-2.5-pro": (150, 2_000_000),
                "gemini-2.0-flash": (2_000, 4_000_000),
                "gemini-2.5-flash": (1_000, 1_000_000),
                "gpt-4o-mini": (5_000, 4_000_000),
                "gpt-4o": (5_000, 800_000),
            }
        elif tier == 2:
            raise ValueError(message.format(tier=2))
        elif tier == 3:
            limits = {
                "gpt-4o-mini": (5_000, 4_000_000),
                "gpt-4o": (5_000, 800_000),
            }
        elif tier == 4:
            limits = {
                "gpt-4o-mini": (10_000, 10_000_000),
                "gpt-4o": (10_000, 2_000_000),
                "gpt-4.1-nano": (10_000, 10_000_000),
                "gpt-4.1-mini": (10_000, 10_000_000),
                "gpt-4.1": (10_000, 2_000_000),
            }
        elif tier == 5:
            limits = {
                "gpt-4o-mini": (30_000, 150_000_000),
                "gpt-4o": (10_000, 30_000_000),
                "gpt-4.1-nano": (30_000, 150_000_000),
                "gpt-4.1-mini": (30_000, 150_000_000),
                "gpt-4.1": (10_000, 30_000_000),
                "gpt-5-mini": (30_000, 180_000_000),
                "gpt-5": (15_000, 40_000_000),
                "gpt-5.2": (15_000, 40_000_000),
            }
        elif tier == AZURE_TIER:
            limits = {
                "gpt-4o-mini": (2_500, 250_000),
                "gpt-4o": (1_800, 300_000),
                "gpt-4.1-mini": (1_105, 1_105_000),
                "gpt-4.1": (250, 250_000),
                "gpt-5-mini": (250, 250_000),
            }
        else:
            raise ValueError(f"Invalid tier: {tier}. Must be between 1 and 5.")

        rate_limits = find_best_match(model, limits)
        if not rate_limits:
            raise ValueError(f"Model {model} is not supported for tier {tier}.")

    request_limit, token_limit = rate_limits
    return ChatRateLimiter(request_limit=request_limit, token_limit=token_limit)


def find_best_match(
    model: str, limits: Mapping[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Find limit to use for model based on the longest prefix match.

    E.g. `gpt-4o-mini-2024-07-18` would match `gpt-4o-mini`.
    """
    matching_prefixes = [prefix for prefix in limits if model.startswith(prefix)]
    if not matching_prefixes:
        return None
    return limits[max(matching_prefixes, key=len)]
