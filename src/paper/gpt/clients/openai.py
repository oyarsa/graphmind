"""OpenAI API client implementation."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable
from typing import Any, ClassVar, cast, override

import backoff
import openai
from openai import NOT_GIVEN, AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from paper.gpt.clients.base import LLMClient
from paper.gpt.models import (
    MODELS_ALLOWED,
    MODELS_ALLOWED_AZURE,
    SearchLevel,
    calc_cost,
)
from paper.gpt.rate_limits import get_api_tier, get_rate_limiter
from paper.gpt.result import GPTResult
from paper.util import ensure_envvar, mustenv

logger = logging.getLogger(__name__)


def prompts_to_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Format system and user prompts as OpenAI-style messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
            api_tier = get_api_tier("azure")
        else:
            api_tier = get_api_tier("openai", base_url)
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
                model_base = model

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
            system_prompt: Text for the system prompt (role: system).
            user_prompt: Text for the user prompt (role: user).
            max_tokens: Maximum number of tokens in the output.
            temperature: Temperature for the model, between 0 and 2. Defaults to 0 to
                try to get consistent outputs from the model.
            seed: Random seed for the request. Defaults to 0 to hopefully be
                reproducible.

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
            cost = calc_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
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
        search_level: SearchLevel | None = None,
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
            {"search_context_size": search_level.value} if is_search else NOT_GIVEN
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
            cost = calc_cost(
                self.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                search_level=search_level,
                used_search=is_search,
            )
        else:
            cost = 0

        try:
            content = completion.choices[0].message.content
        except Exception:
            content = None

        return GPTResult(result=content, cost=cost)

    @staticmethod
    def _fix_params_for_gpt5(params: dict[str, Any]) -> dict[str, Any]:
        """Adjust API params for GPT-5 model restrictions.

        GPT-5 models don't accept:
        - max_tokens=null (must be omitted)
        - temperature=0 (only default value 1 is supported)
        """
        model = params.get("model", "")
        if not model.startswith("gpt-5"):
            return params

        params = params.copy()

        if params.get("max_tokens") is None:
            params.pop("max_tokens", None)

        if params.get("temperature") == 0:
            logger.debug(
                "Model %s only supports temperature=1. Changing to that.", model
            )
            params.pop("temperature", None)

        return params

    @backoff.on_exception(
        backoff.expo,
        (openai.APIError, asyncio.TimeoutError),
        max_tries=5,
        logger=logger,
    )
    async def _call_gpt(self, **chat_params: Any):  # noqa: ANN202
        chat_params = self._fix_params_for_gpt5(chat_params)

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
        chat_params = self._fix_params_for_gpt5(chat_params)

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
