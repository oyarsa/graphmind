"""Google Gemini API client implementation."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, ClassVar, Self, override

import backoff
from google import genai  # type: ignore
from google.genai import errors, types  # type: ignore
from pydantic import BaseModel

from paper.gpt.clients.base import LLMClient
from paper.gpt.models import MODELS_ALLOWED, SearchLevel, calc_cost
from paper.gpt.rate_limits import get_api_tier, get_rate_limiter
from paper.gpt.result import GPTResult
from paper.util import ensure_envvar

logger = logging.getLogger(__name__)


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
        api_tier = get_api_tier("gemini")
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
            cost = calc_cost(
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
        search_level: SearchLevel | None = None,
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

        # Check if grounding was actually used (only charged if grounding metadata present)
        used_grounding = (
            is_search and getattr(completion, "grounding_metadata", None) is not None
        )

        if usage := completion.usage_metadata:
            cost = calc_cost(
                self.model,
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
                search_level=search_level if search_level else SearchLevel.LOW,
                used_search=used_grounding,
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
