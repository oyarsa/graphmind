"""Interact with the OpenAI API."""

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import backoff
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from paper.gpt.model import PromptResult
from paper.util import log_memory_usage
from paper.util.rate_limiter import ChatRateLimiter
from paper.util.serde import (
    Record,
    load_data_jsonl,
    save_data_jsonl,
)

logger = logging.getLogger(__name__)

MODEL_SYNONYMS: Mapping[str, str] = {
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
    "gpt-4o": "gpt-4o-2024-08-06",
}
"""Mapping between short and common model names and their full versioned names."""
MODELS_ALLOWED: Sequence[str] = sorted(MODEL_SYNONYMS.keys() | MODEL_SYNONYMS.values())
"""All allowed model names, including synonyms and full names."""

MODEL_COSTS: Mapping[str, tuple[float, float]] = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
}
"""Cost in $ per 1M tokens: (input cost, output cost).

From https://openai.com/api/pricing/
"""


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


@dataclass(frozen=True)
class GPTResult[T]:
    """Result of a GPT request and its full API cost."""

    result: T
    cost: float


def get_rate_limiter(tier: int, model: str) -> ChatRateLimiter:
    """Get the rate limiter for a specific model based on the API tier.

    Args:
        tier: Tier the organisation is in. Currently supports tier 3 and 4.
        model: API model name. Currently supports 'gpt-4o-mini' and 'gpt-4o'.

    Returns:
        Rate limiter for the model with the correct rate limits for the tier.

    Raises:
        ValueError if tier or model are invalid.
    """
    message = (
        "Tier {tier} limits are not set. Please provide the limits. You can find them on"
        " https://platform.openai.com/settings/organization/limits or using the"
        " `scripts/tools/rate_limits.py` tool"
    )

    # <request_limit, token_limit> per minute
    limits: dict[str, tuple[int, int]]

    if tier == -1:
        rate_limits = (1_000, 1_000_000)
    else:
        if tier == 1:
            raise ValueError(message.format(tier=1))
        if tier == 2:
            raise ValueError(message.format(tier=2))
        if tier == 3:
            limits = {
                "gpt-4o-mini": (5_000, 4_000_000),
                "gpt-4o": (5_000, 800_000),
            }
        elif tier == 4:
            limits = {
                "gpt-4o-mini": (10_000, 10_000_000),
                "gpt-4o": (10_000, 2_000_000),
            }
        elif tier == 5:
            raise ValueError(message.format(tier=5))
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


class ModelClient:
    """Client to communicate with the OpenAI API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        seed: int,
        temperature: float = 0,
        base_url: str | None = None,
        timeout: float = 60,
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
        """
        is_openai = base_url is None or "openai" in base_url
        model = MODEL_SYNONYMS.get(model, model)

        if is_openai and model not in MODELS_ALLOWED:
            raise ValueError(
                f"Invalid model: {model!r}. Should be one of: {MODELS_ALLOWED}."
            )

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.timeout = timeout

        if not is_openai:
            api_tier = -1
        elif api_tier_s := os.getenv("OPENAI_API_TIER"):
            api_tier = int(api_tier_s)
        else:
            logger.warning("OPENAI_API_TIER unset. Defaulting to tier 1.")
            api_tier = 1

        self.rate_limiter = get_rate_limiter(api_tier, model)

    async def run[T: BaseModel](
        self,
        class_: type[T],
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
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
            completion = await self._call_gpt(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=class_,
                seed=self.seed,
                temperature=self.temperature,
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

        return GPTResult(result=completion.choices[0].message.parsed, cost=cost)

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
                return response
        except openai.APIError as e:
            logger.warning("\nCaught an API error: %s", e)
            raise
        except Exception:
            logger.exception("\nCaught non-API error. Returning None.")
            return None


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
        save_data_jsonl(path, result)
        log_memory_usage(path.parent / "memory.txt")
    except Exception:
        logger.exception("Error writing intermediate results to: %s", path)


@dataclass(frozen=True, kw_only=True)
class RemainingItems[T, U]:
    """Contains `done` items loaded from intermediate files and those `remaining`."""

    remaining: list[U]
    done: list[T]


def init_remaining_items[T: Record, U: Record](
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
    output_intermediate_file = output_dir / "results.tmp.jsonl"

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


def get_remaining_items[T: Record, U: Record](
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
