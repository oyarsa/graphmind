"""Abstract base class for LLM clients."""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import BaseModel

from paper.gpt.models import SearchLevel, resolve_model_name
from paper.gpt.tokenizer import prepare_messages
from paper.util import ensure_envvar

if TYPE_CHECKING:
    from paper.gpt.result import GPTResult

logger = logging.getLogger(__name__)


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
        self.model = resolve_model_name(model)
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
        # Import here to avoid circular imports
        from paper.gpt.clients.gemini import GeminiClient
        from paper.gpt.clients.openai import OpenAIClient

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
        # Import here to avoid circular imports
        from paper.gpt.clients.gemini import GeminiClient
        from paper.gpt.clients.openai import OpenAIClient

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
        search_level: SearchLevel | None = None,
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
