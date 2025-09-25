"""Interact with the LLM APIs. Currently supports OpenAI (including Azure) and Gemini."""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    cast,
    overload,
    override,
)

import backoff
import openai
import tiktoken
from google import genai  # type: ignore
from google.genai import errors, types  # type: ignore
from openai import NOT_GIVEN, AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from paper.gpt.model import PromptResult
from paper.types import Identifiable
from paper.util import ensure_envvar, log_memory_usage, mustenv
from paper.util.rate_limiter import ChatRateLimiter
from paper.util.serde import Compress, load_data_jsonl, save_data_jsonl

logger = logging.getLogger(__name__)

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
}
"""Cost in $ per 1M tokens: (input cost, output cost).

From https://openai.com/api/pricing/
"""

MODELS_ALLOWED_AZURE = {"gpt-4o", "gpt-4o-mini"}
"""All allowed model names from the Azure API."""
AZURE_TIER = 10
"""Separate tier for Azure API."""


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate API request based on the model and input/output tokens.

    NB: prompt_tokens/completion_tokens is the name given to input/output tokens in the
    usage object from the OpenAI result.

    Args:
        model: OpenAI model key
        prompt_tokens: the input tokens for the API
        completion_tokens: the output tokens from the API

    Returns:
        The total cost of the request. If the model is invalid, returns 0.
    """
    if model not in MODEL_COSTS:
        return 0

    input_cost, output_cost = MODEL_COSTS[model]
    return prompt_tokens / 1e6 * input_cost + completion_tokens / 1e6 * output_cost


T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, kw_only=True)
class GPTResult(Generic[T_co]):  # noqa: UP046
    """Result of a GPT request and its full API cost."""

    result: T_co
    cost: float

    def map[U](self, func: Callable[[T_co], U]) -> GPTResult[U]:
        """Apply `func` to inner value and return new result."""
        return GPTResult(result=func(self.result), cost=self.cost)

    def then[U](self, other: GPTResult[U]) -> GPTResult[U]:
        """Combine two request costs with the second result."""
        return GPTResult(result=other.result, cost=self.cost + other.cost)

    def bind[U](self, func: Callable[[T_co], GPTResult[U]]) -> GPTResult[U]:
        """Apply monadic function to inner value and sum the costs."""
        return self.then(func(self.result))

    async def abind[U](
        self, func: Callable[[T_co], Awaitable[GPTResult[U]]]
    ) -> GPTResult[U]:
        """Apply monadic function to inner value and sum the costs (async version)."""
        return self.then(await func(self.result))

    async def amap[U](self, func: Callable[[T_co], Awaitable[U]]) -> GPTResult[U]:
        """Apply `func` to inner value and return new result (async version)."""
        return GPTResult(result=await func(self.result), cost=self.cost)

    @staticmethod
    def unit[T](value: T) -> GPTResult[T]:
        """New result with cost 0."""
        return GPTResult(result=value, cost=0)

    def fix[X](self: GPTResult[X | None], default: Callable[[], X] | X) -> GPTResult[X]:
        """Fix the result of the GPTResult by replacing None with the result of `default`.

        Args:
            self: GPTResult to fix containing a value or None.
            default: Function that returns the default value or the value itself.

        Returns:
            GPTResult with the result replaced by the default if it was None.
            If the result is already valid (not None), it returns itself.
        """
        if gpt_is_valid(self):
            return self

        if callable(default):
            value = cast(X, default())
        else:
            value: X = default

        return self.map(lambda _: value)

    @overload
    def lift[U, V](
        self, other: GPTResult[U], func: Callable[[T_co, U], V]
    ) -> GPTResult[V]: ...

    @overload
    def lift[U1, U2, V](
        self,
        other1: GPTResult[U1],
        other2: GPTResult[U2],
        func: Callable[[T_co, U1, U2], V],
    ) -> GPTResult[V]: ...

    @overload
    def lift[U1, U2, U3, V](
        self,
        other1: GPTResult[U1],
        other2: GPTResult[U2],
        other3: GPTResult[U3],
        func: Callable[[T_co, U1, U2, U3], V],
    ) -> GPTResult[V]: ...

    def lift(self, *args: Any, **_: Any) -> GPTResult[Any]:
        """Combine multiple results with an n-ary function."""
        *others, func = args

        values = [self, *others]
        results = (v.result for v in values)
        total_cost = sum(v.cost for v in values)

        return GPTResult(result=func(*results), cost=total_cost)


def gpt_is_valid[T](result: GPTResult[T | None]) -> TypeGuard[GPTResult[T]]:
    """Check if the GPTResult is valid, i.e. has a non-None result."""
    return result.result is not None


def gpt_is_none[T](result: GPTResult[T | None]) -> TypeGuard[GPTResult[None]]:
    """Check if the GPTResult is empty, i.e. has a None result."""
    return result.result is None


def gpt_is_type[T, U](result: GPTResult[T], type_: type[U]) -> TypeGuard[GPTResult[U]]:
    """Check if the GPTResult is of a specific type."""
    return isinstance(result.result, type_)


def gpt_unit[T](value: T) -> GPTResult[T]:
    """Create a unit GPTResult with the given value and cost 0.

    Use this instead of GPTResult.unit for type inference.
    """
    return GPTResult[T].unit(value)


def gpt_sequence[T](results: Iterable[GPTResult[T]]) -> GPTResult[Sequence[T]]:
    """Convert sequence of results to result of sequence, aggregating costs."""
    items: list[T] = []
    total_cost = 0.0

    for result in results:
        items.append(result.result)
        total_cost += result.cost

    return GPTResult(result=tuple(items), cost=total_cost)


def gpr_traverse[T, U](
    results: Iterable[GPTResult[PromptResult[T]]], f: Callable[[T], U]
) -> GPTResult[Sequence[PromptResult[U]]]:
    """Sequence results and map function over values in GPTResult+PromptResult stack."""
    return gpt_sequence(results).map(lambda items: tuple(item.map(f) for item in items))


def gpr_map[T, U](
    result: GPTResult[PromptResult[T]], f: Callable[[T], U]
) -> GPTResult[PromptResult[U]]:
    """Map functions through GPTResult+PromptResult stack."""
    return result.map(lambda r: r.map(f))


def _get_api_tier(provider: str, base_url: str | None = None) -> int:
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
            }
        elif tier == AZURE_TIER:
            limits = {
                "gpt-4o-mini": (2_500, 250_000),
                "gpt-4o": (1_800, 300_000),
            }
        else:
            raise ValueError(f"Invalid tier: {tier}. Must be between 1 and 5.")

        rate_limits = _find_best_match(model, limits)
        if not rate_limits:
            raise ValueError(f"Model {model} is not supported for tier {tier}.")

    request_limit, token_limit = rate_limits
    return ChatRateLimiter(request_limit=request_limit, token_limit=token_limit)


def _find_best_match(
    model: str, limits: Mapping[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Find limit to use for model based on the longest prefix match.

    E.g. `gpt-4o-mini-2024-07-18` would match `gpt-4o-mini`.
    """
    matching_prefixes = [prefix for prefix in limits if model.startswith(prefix)]
    if not matching_prefixes:
        return None
    return limits[max(matching_prefixes, key=len)]


def prepare_messages(
    system_prompt: str, user_prompt: str, max_input_tokens: int | None
) -> tuple[str, str]:
    """Prepare messages for the API call, applying token limits if needed.

    Args:
        system_prompt: Text for the system prompt.
        user_prompt: Text for the user prompt.
        max_input_tokens: Maximum number of input/prompt tokens.

    Returns:
        Tuple of (system, user) prompts.
    """
    if max_input_tokens is None:
        return system_prompt, user_prompt

    system_tokens = count_tokens(system_prompt)
    user_tokens = count_tokens(user_prompt)

    if system_tokens + user_tokens <= max_input_tokens:
        return system_prompt, user_prompt

    available_tokens = max(0, max_input_tokens - system_tokens)
    truncated_user_prompt = truncate_text(user_prompt, available_tokens)

    return system_prompt, truncated_user_prompt


class LLMClient(ABC):
    """ABC for LLM clients."""

    API_KEY_VAR: ClassVar[str]
    """Name of the environment variable holding the API key."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        seed: int,
        temperature: float = 0,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> None:
        """Initialize common LLM client attributes.

        Args:
            api_key: Authentication key.
            model: Model code to use.
            seed: Seed to give the model.
            temperature: How unpredictable the model is.
            timeout: Timeout in seconds for the API calls.
            max_input_tokens: Maximum number of tokens allowed in the input.
            log_exception: If True, log full traceback for non-API exceptions.
        """
        self.api_key = api_key
        self.model = MODEL_SYNONYMS.get(model, model)
        self.seed = seed
        self.temperature = temperature
        self.timeout = timeout
        self.max_input_tokens = max_input_tokens
        self._calls_made = 0
        self._tokens_used = 0

        if log_exception is not None:
            self.should_log_exception = log_exception
        else:
            self.should_log_exception = os.getenv("LOG_EXCEPTION", "0") == "1"

    def _log_exception(self, msg: str, exc: Exception) -> None:
        """Log exception with appropriate level based on configuration."""
        log = logger.exception if self.should_log_exception else logger.warning
        log(msg, exc)

    async def _prepare_prompts(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, str]:
        """Prepare prompts applying token limits if needed."""
        return await asyncio.to_thread(
            prepare_messages, system_prompt, user_prompt, self.max_input_tokens
        )

    @classmethod
    def new(
        cls,
        *,
        model: str,
        seed: int,
        api_key: str | None = None,
        temperature: float = 0,
        base_url: str | None = None,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> LLMClient:
        """Create a client for Gemini, OpenAI or other compatible APIs.

        Uses separate implementations for Gemini and OpenAI. If the model contains,
        'gemini', the Gemini client is used. Everything else goes to the OpenAI-compatible
        client.

        You can also use this with Ollama, OpenRouter and other OpenAI-compatible APIs.
        They must be compatible with Structured Outputs.

        Args:
            model: Model code to use. See your API documentation for the name.
            seed: Seed to give the model.
            api_key: Authentication key. If absent, we'll fetch from the implementation's
                environment variable (e.g. OPENAI_API_KEY or GEMINI_API_KEY).
            temperature: How unpredictable the model is. Set this 0 to be as
                deterministic as possible, but it's still not guaranteed.
            base_url: URL of the API being used. If not provided, use OpenAI.
            timeout: Timeout in seconds for the API calls.
            max_input_tokens: Maximum number of tokens allowed in the input.
                If set, will truncate the `user_prompt` if it exceeds this limit.
            log_exception: If True, log full traceback for non-API exceptions. If False,
                log the exception description as a warning. If None, get value from
                the `LOG_EXCEPTION` environment variable (1 or 0), defaulting to 0.
        """
        impl_class = GeminiClient if "gemini" in model else OpenAIClient

        return impl_class(
            api_key=api_key or ensure_envvar(impl_class.API_KEY_VAR),
            model=model,
            seed=seed,
            temperature=temperature,
            base_url=base_url,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
        )

    @classmethod
    def new_env(
        cls,
        model: str,
        seed: int,
        temperature: float = 0,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> LLMClient:
        """Create new client for Gemini, OpenAI, etc. from environment variables.

        Uses separate implementations for Gemini and OpenAI. If the model contains,
        'gemini', the Gemini client is used. Everything else goes to the OpenAI-compatible
        client.

        You can also use this with Ollama, OpenRouter and other OpenAI-compatible APIs.
        They must be compatible with Structured Outputs.

        See also: `GeminiClient.from_env`, `OpenAIClient.from_env`
        """
        impl_class = GeminiClient if "gemini" in model else OpenAIClient
        return impl_class.from_env(
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
        )

    @classmethod
    @abstractmethod
    def from_env(
        cls,
        model: str,
        seed: int,
        temperature: float = 0,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> Self:
        """Create new client from environment variables."""
        ...

    @abstractmethod
    async def run[T: BaseModel](
        self,
        class_: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[T | None]:
        """Run the query and return a parsed object of `class_`."""

    @abstractmethod
    async def plain(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        search_level: Literal["low", "medium", "high"] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[str | None]:
        """Run the GPT query and return plain text output."""

    @property
    def calls_made(self) -> int:
        """How many calls were made over the lifetime of this client."""
        return self._calls_made

    @property
    def tokens_used(self) -> int:
        """How many tokens were used over the lifetime of this client."""
        return self._tokens_used


class OpenAIClient(LLMClient):
    """Client to communicate with the OpenAI API."""

    API_KEY_VAR: ClassVar[str] = "OPENAI_API_KEY"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        seed: int,
        temperature: float = 0,
        base_url: str | None = None,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> None:
        """Create client for OpenAI-compatible APIs.

        You can also use this with Ollama, OpenRouter and other compatible APIs.
        They must be compatible with Structured Outputs.

        Args:
            api_key: Authentication key. For Ollama, this can be anything, but it must
                be non-empty.
            model: Model code to use. See your API documentation for the name.
            seed: Seed to give the model.
            temperature: How unpredictable the model is. Set this 0 to be as
                deterministic as possible, but it's still not guaranteed.
            base_url: URL of the API being used. If not provided, use OpenAI.
            timeout: Timeout in seconds for the API calls.
            max_input_tokens: Maximum number of tokens allowed in the input.
                If set, will truncate the `user_prompt` if it exceeds this limit.
            log_exception: If True, log full traceback for non-API exceptions. If False,
                log the exception description as a warning. If None, get value from
                the `LOG_EXCEPTION` environment variable (1 or 0), defaulting to 0.
        """
        # Initialize common attributes first
        super().__init__(
            api_key=api_key,
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
        )

        self.base_url = base_url
        is_openai = base_url is None
        is_azure = base_url is not None and "azure" in base_url

        # Azure doesn't use model synonyms
        if is_azure:
            self.model = model  # Override the synonym resolution

        if is_openai and self.model not in MODELS_ALLOWED:
            raise ValueError(
                f"Invalid OpenAI model: '{self.model}'. Should be one of: {MODELS_ALLOWED}."
            )

        if is_azure:
            if model not in MODELS_ALLOWED_AZURE:
                raise ValueError(
                    f"Invalid Azure model: '{model}'. Should be one of:"
                    f" {MODELS_ALLOWED_AZURE}."
                )

            assert base_url, "Base URL must be non-empty for Azure"
            self.client = AsyncAzureOpenAI(
                azure_endpoint=base_url,
                api_version="2024-12-01-preview",
                api_key=api_key,
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Determine API tier and create rate limiter
        if is_azure:
            api_tier = _get_api_tier("azure")
        else:
            api_tier = _get_api_tier("openai", base_url)
        self.rate_limiter = get_rate_limiter(api_tier, self.model)

    @classmethod
    def from_env(
        cls,
        model: str,
        seed: int,
        temperature: float = 0,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> OpenAIClient:
        """Create new OpenAIClient based on environment variables.

        This has two modes: Azure or standard OpenAI. If the environment variable
        `USE_AZURE` is 1, we use it, otherwise we use the standard client.

        The standard client requires the `OPENAI_API_KEY` environment variable.
        Optionally, `OPENAI_BASE_URL` can also be set.

        The Azure client requires both `AZURE_BASE_URL` and `AZURE_API_KEY`. It's
        possible that the base URL differs between models, so we replace `{{model}}` with
        the parameter.
        """
        use_azure = os.getenv("USE_AZURE", "0") == "1"

        if use_azure:
            # Azure requires simplified model names without version suffixes
            if model.startswith("gpt-4o-mini"):
                model_base = "gpt-4o-mini"
            elif model.startswith("gpt-4o"):
                model_base = "gpt-4o"
            else:
                raise ValueError(f"Invalid Azure model: {model}")

            if model_base not in MODELS_ALLOWED_AZURE:
                raise ValueError(f"Invalid Azure model: {model}")

            env = mustenv("AZURE_BASE_URL", "AZURE_API_KEY")
            base_url = env["AZURE_BASE_URL"].replace("{{model}}", model_base)
            api_key = env["AZURE_API_KEY"]
            model = model_base  # Use simplified name for Azure
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = ensure_envvar("OPENAI_API_KEY")

        return cls(
            api_key=api_key,
            model=model,
            seed=seed,
            temperature=temperature,
            base_url=base_url,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
        )

    @override
    async def run[T: BaseModel](
        self,
        class_: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[T | None]:
        """Run the GPT query and return a parsed object of `class_`.

        Uses Structured Outputs to get a valid object. See also:
        https://platform.openai.com/docs/guides/structured-outputs

        Args:
            class_: The class to parse the Structured Outputs. Must be a Pydantic
                BaseModel. Note that this is the class (type) itself, not an instance.
            client: The OpenAI API client (e.g. openai.OpenAI or openai.AzureOpenAI).
            system_prompt: Text for the system prompt (role: system).
            user_prompt: Text for the user prompt (role: user).
            model: Full GPT model name. See `MODELS_ALLOWED`.
            seed: Random seed for the request. Defaults to 0 to hopefully be
                reproducible.
            temperature: Temperature for the model, between 0 and 2. Defaults to 0 to
                try to get consistent outputs from the model.
            max_tokens: Maximum number of tokens in the output.

        Returns:
            Result with the cost for the request and the result object parsed. If there
            was an error with the request (mainly when making the request itself or
            parsing the result), the object is None. The cost is provided either way.

        Raises:
            ValueError: if the `model` is invalid (see `MODELS_ALLOWED`).
        """

        try:
            system_prompt, user_prompt = await self._prepare_prompts(
                system_prompt, user_prompt
            )
            completion = await self._call_gpt(
                model=self.model,
                messages=prompts_to_messages(system_prompt, user_prompt),
                response_format=class_,
                seed=seed if seed is not None else self.seed,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("Error when calling OpenAI. Gave up on retrying")
            completion = None

        if completion is None:
            return GPTResult(result=None, cost=0)

        if usage := completion.usage:
            cost = _calc_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
        else:
            cost = 0

        choice = completion.choices[0]
        return GPTResult(result=choice.message.parsed, cost=cost)

    @override
    async def plain(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        search_level: Literal["low", "medium", "high"] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[str | None]:
        """Run the GPT query and return plain text output.

        Uses the standard completion API.

        Args:
            system_prompt: Text for the system prompt (role: system).
            user_prompt: Text for the user prompt (role: user).
            max_tokens: Maximum number of tokens in the output.
            search_level: If given, use the web search tool with this context size.
                Search results will be given as part of the textual response only.
                See https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat
                for more information.
            temperature: Temperature override for this specific request.
            seed: Seed override for this specific request.

        Returns:
            Result with the cost for the request and the plain text response. If there
            was an error with the request, the result is None. The cost is provided
            either way.
        """
        is_search = search_level is not None

        # Remove temperature and seed for web search models, as they aren't supported.
        final_temperature = (
            (temperature if temperature is not None else self.temperature)
            if not is_search
            else NOT_GIVEN
        )
        final_seed = (
            (seed if seed is not None else self.seed) if not is_search else NOT_GIVEN
        )
        search_options = (
            {"search_context_size": search_level} if is_search else NOT_GIVEN
        )

        try:
            system_prompt, user_prompt = await self._prepare_prompts(
                system_prompt, user_prompt
            )
            completion = await self._call_gpt_plain(
                model=self.model,
                messages=prompts_to_messages(system_prompt, user_prompt),
                seed=final_seed,
                temperature=final_temperature,
                max_tokens=max_tokens,
                web_search_options=search_options,
            )
        except Exception:
            logger.exception("Error when calling OpenAI. Gave up on retrying")
            return GPTResult(result=None, cost=0)

        if completion is None:
            return GPTResult(result=None, cost=0)

        if usage := completion.usage:
            cost = _calc_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
        else:
            cost = 0

        try:
            content = completion.choices[0].message.content
        except Exception:
            content = None

        return GPTResult(result=content, cost=cost)

    @backoff.on_exception(
        backoff.expo,
        (openai.APIError, asyncio.TimeoutError),
        max_tries=5,
        logger=logger,
    )
    async def _call_gpt(self, **chat_params: Any):  # noqa: ANN202
        try:
            async with self.rate_limiter.limit(**chat_params) as update_usage:
                response = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(**chat_params),
                    timeout=self.timeout,
                )
                await update_usage(response)
                self._calls_made += 1
                if usage := response.usage:
                    self._tokens_used += usage.total_tokens
                return response
        except openai.APIError as e:
            logger.warning("API error: %s", e)
            raise
        except Exception as e:
            logger.warning("Non-API error: %s", e)
            return None

    @backoff.on_exception(
        backoff.expo,
        (openai.APIError, asyncio.TimeoutError),
        max_tries=5,
        logger=logger,
    )
    async def _call_gpt_plain(self, **chat_params: Any) -> ChatCompletion | None:
        """Call OpenAI API to generate plain text completions.

        Args:
            **chat_params: Parameters to pass to the chat.completions.create method.

        Returns:
            The ChatCompletion response object or None if there was an error.

        Raises:
            openai.APIError: If there was an API error that should be retried.
        """
        try:
            async with self.rate_limiter.limit(**chat_params) as update_usage:
                # Create the task with proper type annotation
                task = cast(
                    Awaitable[ChatCompletion],
                    self.client.chat.completions.create(**chat_params),
                )
                response = await asyncio.wait_for(task, timeout=self.timeout)
                await update_usage(response)
                return response
        except openai.APIError as e:
            logger.warning("API error: %s", e)
            raise
        except Exception as e:
            logger.warning("Non-API error: %s", e)
            return None


def prompts_to_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Format system and user prompts as OpenAI-style messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class GeminiClient(LLMClient):
    """Client to communicate with the Google Gemini API."""

    API_KEY_VAR: ClassVar[str] = "GEMINI_API_KEY"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        seed: int,
        temperature: float = 0,
        base_url: str | None = None,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
        thinking_budget: int | None = None,
        include_thoughts: bool | None = None,
    ) -> None:
        """Create client for the Google Gemini API.

        Args:
            api_key: Authentication key.
            model: Model code to use. See your API documentation for the name.
            seed: Seed to give the model.
            temperature: How unpredictable the model is. Set this 0 to be as
                deterministic as possible, but it's still not guaranteed.
            base_url: Ignored, as the Gemini client doesn't have a use for it.
            timeout: Timeout in seconds for the API calls.
            max_input_tokens: Maximum number of tokens allowed in the input.
                If set, will truncate the `user_prompt` if it exceeds this limit.
            log_exception: If True, log full traceback for non-API exceptions. If False,
                log the exception description as a warning. If None, get value from
                the `LOG_EXCEPTION` environment variable (1 or 0), defaulting to 0.
            thinking_budget: Number of tokens a model can use to think. To disable
                thinking entirely, set to 0. If unset, uses the model default. If set,
                must be between 0 and 24576.
            include_thoughts: Whether to include thoughts in the response. If unset,
                uses the model default.

        `thinking_budget` and `include_thoughts` are NOT ignored if the model does not
        support thinking. Attempting to set them with non-thinking model will result in
        an API error.

        Raises:
            ValueError: if the model is invalid or if `thinking_budget` is out of range.
        """
        # Initialize common attributes first
        super().__init__(
            api_key=api_key,
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
        )

        if self.model not in MODELS_ALLOWED:
            raise ValueError(
                f"Invalid model: {self.model!r}. Should be one of: {MODELS_ALLOWED}."
            )

        self.client = genai.Client(api_key=api_key)
        self.base_url = base_url
        self.include_thoughts = include_thoughts

        if thinking_budget is not None and not (0 <= thinking_budget <= 24576):
            raise ValueError("thinking_budget must be in [0, 24576]")
        self.thinking_budget = thinking_budget

        # Get API tier and create rate limiter
        api_tier = _get_api_tier("gemini")
        self.rate_limiter = get_rate_limiter(api_tier, self.model)

    @classmethod
    def from_env(
        cls,
        model: str,
        seed: int,
        temperature: float = 0,
        timeout: float = 60,
        max_input_tokens: int | None = 90_000,
        log_exception: bool | None = None,
    ) -> Self:
        """Create new GeminiClient with a key from environment variables.

        - GEMINI_API_KEY: key to the API.
        - INCLUDE_THOUGHTS: 1 to include thoughts, 0 to exclude. If unset, use model
            default.
        - THINKING_BUDGET: number of tokens to use for thinking. Use 0 to disable
            thinking. If set, must in [0, 24576]. If unset, use model default.

        Args:
            model: Model code to use. See your API documentation for the name.
            seed: Seed to give the model.
            temperature: How unpredictable the model is. Set this 0 to be as
                deterministic as possible, but it's still not guaranteed.
            timeout: Timeout in seconds for the API calls.
            max_input_tokens: Maximum number of tokens allowed in the input.
                If set, will truncate the `user_prompt` if it exceeds this limit.
            log_exception: If True, log full traceback for non-API exceptions. If False,
                log the exception description as a warning. If None, get value from
                the `LOG_EXCEPTION` environment variable (1 or 0), defaulting to 0.

        Raises:
            ValueError: if `model` is invalid; or if THINKING_BUDGET is given but isn't
                an integer, or if it's out of range.
        """
        api_key = ensure_envvar(cls.API_KEY_VAR)

        include_thoughts: bool | None = None
        thinking_budget: int | None = None

        match os.getenv("INCLUDE_THOUGHTS"):
            case "1":
                include_thoughts = True
            case "0":
                include_thoughts = False
            case None | "":  # Unset
                pass
            case _:
                raise ValueError("INCLUDE_THOUGHTS must be unset, 0 or 1")

        if (thinking_budget_env := os.getenv("THINKING_BUDGET")) is not None:
            try:
                thinking_budget = int(thinking_budget_env)
            except ValueError as e:
                raise ValueError("THINKING_BUDGET must be an integer") from e

        return cls(
            api_key=api_key,
            model=model,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            max_input_tokens=max_input_tokens,
            log_exception=log_exception,
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
        )

    @override
    async def run[T: BaseModel](
        self,
        class_: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[T | None]:
        """Run the query and return a parsed object of `class_`.

        Uses Structured Outputs to get a valid object. See also:
        https://ai.google.dev/gemini-api/docs/structured-output?lang=python#supply-schema-in-config

        Args:
            class_: The class to parse the Structured Outputs. Must be a Pydantic
                BaseModel. Note that this is the class (type) itself, not an instance.
            system_prompt: Text for the system prompt (role: system).
            user_prompt: Text for the user prompt (role: user).
            max_tokens: Maximum number of tokens in the output.
            temperature: Temperature override for this specific request.
            seed: Seed override for this specific request.

        Returns:
            Result with the cost for the request and the result object parsed. If there
            was an error with the request (mainly when making the request itself or
            parsing the result), the object is None. The cost is provided either way.
        """

        system_prompt, user_prompt = await self._prepare_prompts(
            system_prompt, user_prompt
        )
        try:
            completion = await self._call_api(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=class_,
                    system_instruction=system_prompt,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                    seed=seed if seed is not None else self.seed,
                    max_output_tokens=max_tokens,
                    thinking_config=self._thinking_config(),
                ),
            )
        except Exception:
            logger.exception("Error when calling the API. Gave up on retrying")
            completion = None

        if completion is None:
            return GPTResult(result=None, cost=0)

        if usage := completion.usage_metadata:
            cost = _calc_cost(
                self.model,
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
            )
        else:
            cost = 0

        parsed: Any = completion.parsed  # type: ignore
        if not isinstance(parsed, class_):  # Invalid structured output type
            return GPTResult(result=None, cost=cost)

        return GPTResult(result=parsed, cost=cost)

    @override
    async def plain(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        search_level: Literal["low", "medium", "high"] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GPTResult[str | None]:
        """Run the query and return plain text output.

        Uses the standard completion API.

        Args:
            system_prompt: Text for the system prompt (role: system).
            user_prompt: Text for the user prompt (role: user).
            max_tokens: Maximum number of tokens in the output.
            search_level: If given, use the web search tool with this context size.
                Gemini doesn't support specifying the level, so any setting will have
                the same effect.
            temperature: Temperature override for this specific request.
            seed: Seed override for this specific request.

        Returns:
            Result with the cost for the request and the plain text response. If there
            was an error with the request, the result is None. The cost is provided
            either way.
        """
        is_search = search_level is not None
        system_prompt, user_prompt = await self._prepare_prompts(
            system_prompt, user_prompt
        )

        try:
            completion = await self._call_api(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    seed=seed if seed is not None else self.seed,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                    max_output_tokens=max_tokens,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                    if is_search
                    else None,
                    thinking_config=self._thinking_config(),
                ),
            )
        except Exception:
            logger.exception("Error when calling the API. Gave up on retrying")
            return GPTResult(result=None, cost=0)

        if completion is None:
            return GPTResult(result=None, cost=0)

        if usage := completion.usage_metadata:
            cost = _calc_cost(
                self.model,
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
            )
        else:
            cost = 0

        try:
            content = _gemini_content(completion)
        except Exception:
            content = None

        return GPTResult(result=content, cost=cost)

    def _thinking_config(self) -> types.ThinkingConfig | None:
        """Create config if both `thinking_budget` and `include_thoughts` are set."""
        if self.thinking_budget is not None or self.include_thoughts is not None:
            return types.ThinkingConfig(
                include_thoughts=self.include_thoughts,
                thinking_budget=self.thinking_budget,
            )
        else:
            return None

    @backoff.on_exception(
        backoff.expo,
        (errors.APIError, asyncio.TimeoutError),
        max_tries=5,
        logger=logger,
    )
    async def _call_api(
        self, **chat_params: Any
    ) -> types.GenerateContentResponse | None:
        """Call OpenAI API to generate plain text completions.

        Args:
            **chat_params: Parameters to pass to the chat.completions.create method.

        Returns:
            The GenerateContentResponse response object or None if there was an error.

        Raises:
            google.genai.errors.APIError: If there was an API error that should be
            retried.
        """
        try:
            async with self.rate_limiter.limit(**chat_params) as update_usage:
                # Create the task with proper type annotation
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(**chat_params),
                    timeout=self.timeout,
                )
                await update_usage(response)
                self._calls_made += 1
                if (usage := response.usage_metadata) and usage.total_token_count:
                    self._tokens_used += usage.total_token_count
                return response
        except errors.APIError as e:
            logger.warning("API error: %s", e)
            raise
        except Exception as e:
            logger.warning("Non-API error: %s", e)
            return None


def _gemini_content(completion: types.GenerateContentResponse) -> str:
    """Get full content text from Gemini completion."""
    if (
        (candidates := completion.candidates)
        and (candidate := candidates[0])
        and (content := candidate.content)
        and (parts := content.parts)
    ):
        return "\n".join(each.text for each in parts if each.text)

    return ""


async def append_intermediate_result_async[T: BaseModel](
    path: Path, result: PromptResult[T]
) -> None:
    """Async wrapper for append_intermediate_result.

    Save result to intermediate file by appending object to JSON Lines file.
    Runs the sync function in a thread to avoid blocking the event loop.

    All `Exception`s generated by file IO and JSON validation are caught and logged.

    Args:
        path:
            Path to intermediate output file. Used to first read the existing items,
            then to save the new one. An empty one will be created if it doesn't exist.
        result:
            Item to be saved.
    """
    await asyncio.to_thread(append_intermediate_result, path, result)


def append_intermediate_result[T: BaseModel](
    path: Path, result: PromptResult[T]
) -> None:
    """Save result to intermediate file by appending object to JSON Lines file.

    All `Exception`s generated by file IO and JSON validation are caught and logged.

    Args:
        path:
            Path to intermediate output file. Used to first read the existing items,
            then to save the new one. An empty one will be created if it doesn't exist.
        result:
            Item to be saved.
    """
    try:
        save_data_jsonl(path, result, compress=Compress.ZSTD)
        log_memory_usage(path.parent / "memory.txt")
    except Exception:
        logger.exception("Error writing intermediate results to: %s", path)


@dataclass(frozen=True, kw_only=True)
class RemainingItems[T, U]:
    """Contains `done` items loaded from intermediate files and those `remaining`."""

    remaining: Sequence[U]
    done: Sequence[T]


def init_remaining_items[T: Identifiable, U: Identifiable](
    continue_type_: type[T],
    output_dir: Path,
    continue_papers_file: Path | None,
    input_data: Sequence[U],
    continue_: bool = False,
) -> tuple[Path, RemainingItems[PromptResult[T], U]]:
    """Initialise paper processing by handling already processed files.

    Creates output directory and checks intermediate results file to determine which
    papers still need processing. Returns files and remaining items needed for further
    processing.

    See also `get_remaining_items`.

    Args:
        continue_type_: Type of the contents of the remaining data.
        output_dir: Directory where results will be stored.
        continue_papers_file: File containing list of previously items papers.
        input_data: Items to process.
        continue_: If True, skips previously processed items.

    Returns:
        Tuple containing:
            - Path to intermediate results file.
            - RemainingItems containing processed and unprocessed papers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.jsonl.zst"

    papers_remaining = get_remaining_items(
        continue_type_,
        output_intermediate_file,
        continue_papers_file,
        input_data,
        continue_,
    )

    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return output_intermediate_file, papers_remaining

    if continue_:
        logger.info(
            "Skipping %d items from the `continue` file.", len(papers_remaining.done)
        )

    return output_intermediate_file, papers_remaining


def get_remaining_items[T: Identifiable, U: Identifiable](
    continue_type_: type[T],
    output_intermediate_file: Path,
    continue_papers_file: Path | None,
    original: Sequence[U],
    continue_: bool,
) -> RemainingItems[PromptResult[T], U]:
    """Split items that were previously processed from this run's input list.

    Loads data from the intermediate file, then removes the items from the input list
    that appear there. The check is done by the record `id`. The existing items are
    returned as `done`, and the ones left to process as `remaining`. The `done` list
    only contains values in the input data, not all values in the intermediate file.

    Args:
        continue_type_: Pydantic type for the output items that will be read.
        output_intermediate_file: File that stores the processed output.
        continue_papers_file: File with the previously processed items. If this is None
            and `output_intermediate_file` exists, it will be set to that.
        original: Items read for the original dataset file.
        continue_: If true, uses the previous results.

    Returns:
        Done and remaining items as separated lists.
    """
    if continue_papers_file is None and output_intermediate_file.is_file():
        continue_papers_file = output_intermediate_file

    if not continue_:
        if continue_papers_file and continue_papers_file.exists():
            rotate_path(continue_papers_file)
            logger.info(
                "Rotating existing intermediate result file and creating new one."
            )
        return RemainingItems(remaining=list(original), done=[])

    continue_papers: Sequence[PromptResult[T]] = []
    if continue_papers_file:
        logger.info("Continuing items from: %s", continue_papers_file)
        try:
            continue_papers = load_data_jsonl(
                continue_papers_file, PromptResult[continue_type_]
            )
        except Exception:
            logger.exception("Error reading previous files")

    continue_paper_ids = {paper.item.id for paper in continue_papers}
    # Split into papers that _are_ in the continue file.
    done = [
        next(c for c in continue_papers if c.item.id == paper.id)
        for paper in original
        if paper.id in continue_paper_ids
    ]
    # And those that _are not_.
    remaining = [paper for paper in original if paper.id not in continue_paper_ids]

    return RemainingItems(remaining=remaining, done=done)


def rotate_path(path: Path) -> None:
    """Rotate a path by finding the next available numeric suffix.

    If path exists, rename it to path.N where N is the next available number.

    Args:
        path: Path to the file or directory to rotate.
        The original path (not the rotated path).
    """
    if not path.exists():
        # Path doesn't exist, so no rotation needed
        return

    # Find the next available rotation number
    rotation = 0
    while path.with_name(f"{path.name}.{rotation}").exists():
        rotation += 1

    # Rename the original file to use the next available number
    path.rename(path.with_name(f"{path.name}.{rotation}"))


_TOKENISER = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in `text`."""
    return len(safe_tokenise(text))


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        The truncated text.
    """
    tokens = safe_tokenise(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return _TOKENISER.decode(truncated_tokens)


def safe_tokenise(text: str) -> list[int]:
    """Tokenise `text` using the GPT tokeniser. Treats special tokens as regular text."""
    return _TOKENISER.encode(text, disallowed_special=())
