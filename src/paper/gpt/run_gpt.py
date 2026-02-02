"""Interact with the LLM APIs. Currently supports OpenAI (including Azure) and Gemini.

This module re-exports from the following submodules for backwards compatibility:
- models: Model definitions, costs, and search level configuration
- result: GPTResult monad and related utility functions
- rate_limits: Rate limiting configuration
- tokenizer: Token counting and text truncation utilities
- intermediate: Intermediate result file handling
- clients: LLM client implementations (OpenAI, Gemini)
"""

from __future__ import annotations

# Re-export from clients
from paper.gpt.clients import (
    GeminiClient,
    LLMClient,
    OpenAIClient,
)
from paper.gpt.clients.openai import prompts_to_messages

# Re-export from intermediate
from paper.gpt.intermediate import (
    RemainingItems,
    append_intermediate_result,
    append_intermediate_result_async,
    get_remaining_items,
    init_remaining_items,
    rotate_path,
)

# Re-export PromptResult from model (for backwards compatibility)
from paper.gpt.model import PromptResult

# Re-export from models
from paper.gpt.models import (
    AZURE_TIER,
    MODEL_COSTS,
    MODEL_SEARCH_COSTS,
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    MODELS_ALLOWED_AZURE,
    SearchLevel,
    resolve_model_name,
)
from paper.gpt.models import (
    calc_cost as _calc_cost,
)

# Re-export from rate_limits
from paper.gpt.rate_limits import (
    find_best_match as _find_best_match,
)
from paper.gpt.rate_limits import (
    get_api_tier as _get_api_tier,
)
from paper.gpt.rate_limits import (
    get_rate_limiter,
)

# Re-export from result
from paper.gpt.result import (
    GPTResult,
    gpr_map,
    gpr_traverse,
    gpt_is_none,
    gpt_is_type,
    gpt_is_valid,
    gpt_sequence,
    gpt_unit,
)

# Re-export from tokenizer
from paper.gpt.tokenizer import (
    count_tokens,
    prepare_messages,
    safe_tokenise,
    truncate_text,
)

__all__ = [
    "AZURE_TIER",
    "MODELS_ALLOWED",
    "MODELS_ALLOWED_AZURE",
    "MODEL_COSTS",
    "MODEL_SEARCH_COSTS",
    "MODEL_SYNONYMS",
    "GPTResult",
    "GeminiClient",
    "LLMClient",
    "OpenAIClient",
    "PromptResult",
    "RemainingItems",
    "SearchLevel",
    "_calc_cost",
    "_find_best_match",
    "_get_api_tier",
    "append_intermediate_result",
    "append_intermediate_result_async",
    "count_tokens",
    "get_rate_limiter",
    "get_remaining_items",
    "gpr_map",
    "gpr_traverse",
    "gpt_is_none",
    "gpt_is_type",
    "gpt_is_valid",
    "gpt_sequence",
    "gpt_unit",
    "init_remaining_items",
    "prepare_messages",
    "prompts_to_messages",
    "resolve_model_name",
    "rotate_path",
    "safe_tokenise",
    "truncate_text",
]
