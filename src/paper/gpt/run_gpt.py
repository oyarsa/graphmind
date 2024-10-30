import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import backoff
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter

from paper.rate_limiter import ChatRateLimiter

logger = logging.getLogger("paper.gpt.run_gpt")

MODEL_SYNONYMS = {
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
    "gpt-4o": "gpt-4o-2024-08-06",
}
# Include the synonyms and their keys in the allowed models
MODELS_ALLOWED = sorted(MODEL_SYNONYMS.keys() | MODEL_SYNONYMS.values())

# Cost in $ per 1M tokens: (input cost, output cost)
# From https://openai.com/api/pricing/
MODEL_COSTS = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
}

rate_limiters: dict[str, Any] = {
    "gpt-4o-mini": ChatRateLimiter(request_limit=5_000, token_limit=4_000_000),
    "gpt-4o": ChatRateLimiter(request_limit=5_000, token_limit=800_000),
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate API request based on the model and input/output tokens.

    NB: prompt_tokens/completion_tokens is the name given to input/output tokens in the
    usage object from the OpenAI result.

    Args:
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


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    user: str


class PromptResult[T](BaseModel):
    model_config = ConfigDict(frozen=True)

    item: T
    prompt: Prompt


@backoff.on_exception(backoff.expo, openai.APIError, max_tries=5, logger=logger)
async def _call_gpt(
    rate_limiter: Any, client: AsyncOpenAI, chat_params: dict[str, Any]
):
    try:
        async with rate_limiter.limit(**chat_params):
            return await client.beta.chat.completions.parse(**chat_params)
    except openai.APIError as e:
        logger.warning("\nCaught an API error: %s", e)
        raise
    except Exception as e:
        logger.warning("\nCaught non-API error. Returning None: %s", e)
        return None


async def run_gpt[T: BaseModel](
    class_: type[T],
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    seed: int = 0,
    temperature: float = 0,
) -> GPTResult[T | None]:
    """Run the GPT query and return a parsed object of `class_` using Structured Outputs.

    See also: https://platform.openai.com/docs/guides/structured-outputs

    Args:
        class_: The class to parse the Structured Outputs. Must be a Pydantic BaseModel.
            Note that this is the class (type) itself, not an instance.
        client: The OpenAI API client (e.g. openai.OpenAI or openai.AzureOpenAI).
        system_prompt: Text for the system prompt (role: system).
        user_prompt: Text for the user prompt (role: user).
        model: Full GPT model name. See `MODELS_ALLOWED`.
        seed: Random seed for the request. Defaults to 0 to hopefully be reproducible.
        temperature: Temperature for the model, between 0 and 2. Defaults to 0 to try
            to get consistent outputs from the model.

    Returns:
        Result with the cost for the request and the result object parsed. If there
        was an error with the request (mainly when making the request itself or parsing
        the result), the object is None. The cost is provided either way.

    Raises:
        ValueError: if the `model` is invalid (see `MODELS_ALLOWED`).
        RetryError: if the the client should retry the request
    """
    if model not in MODELS_ALLOWED:
        raise ValueError(
            f"Invalid model: {model!r}. Should be one of: {MODELS_ALLOWED}."
        )

    chat_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": class_,
        "seed": seed,
        "temperature": temperature,
    }
    rate_limiter = None
    for limit_model, limiter in rate_limiters.items():
        if model.startswith(limit_model):
            rate_limiter = limiter
    assert rate_limiter is not None

    try:
        completion = await _call_gpt(rate_limiter, client, chat_params)
    except Exception:
        logger.exception("Error when calling OpenAI. Gave up on retrying")
        completion = None

    if completion is None:
        return GPTResult(result=None, cost=0)

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed

    return GPTResult(result=parsed, cost=cost)


def append_intermediate_result[T: BaseModel](
    type_: type[T], path: Path, result: PromptResult[T]
) -> None:
    """Save result to intermediate file.

    How it works: we try to load the existing list of items from the JSON file. If it
    doesn't exist, we start with an empty list. We add the new item to the list, then
    save it to the same file.

    All `Exception`s generated by file IO and JSON validation are caught and logged.

    Args:
        type_:
            Class object of the Pydantic BaseModel representing the data. Needs to be
            provided provided so we can initialise the TypeAdapter, as the generic
            type is lost at runtime.
        path:
            Path to intermediate output file. Used to first read the existing items,
            then to save the new one. An empty one will be created if it doesn't exist.
        result:
            Item to be saved.

    """
    result_adapter = TypeAdapter(list[PromptResult[type_]])

    previous = []
    try:
        previous = result_adapter.validate_json(path.read_bytes())
    except FileNotFoundError:
        # It's fine if the file didn't exist previously. We'll create a new one now.
        pass
    except Exception:
        logger.exception("Error reading intermediate result file: %s", path)

    previous.append(result)
    try:
        path.write_bytes(result_adapter.dump_json(previous, indent=2))
    except Exception:
        logger.exception("Error writing intermediate results to: %s", path)


class HasId(Protocol):
    @property
    def id(self) -> int: ...


def get_remaining_items[T: HasId, U: HasId](
    continue_type_: type[T],
    output_intermediate_file: Path,
    continue_papers_file: Path | None,
    original: Sequence[U],
) -> list[U]:
    """Remove items that were previously processed from this run's input list.

    Args:
        continue_type_: Pydantic type for the output items that will be read.
        output_intermediate_file: File that stores the processed output.
        continue_papers_file: File with the previously processed items. If this is None
            and `output_intermediate_file` exists, it will be set to that.
        original: Items read for the original dataset file.

    Returns:
        Remaining items to be processed.
    """
    if continue_papers_file is None and output_intermediate_file.is_file():
        continue_papers_file = output_intermediate_file

    continue_papers: list[PromptResult[T]] = []
    if continue_papers_file:
        logger.info("Continuing items from: %s", continue_papers_file)
        try:
            continue_papers = TypeAdapter(
                list[PromptResult[continue_type_]]
            ).validate_json(continue_papers_file.read_bytes())
        except Exception:
            logger.exception("Error reading previous files")

    continue_paper_ids = {paper.item.id for paper in continue_papers}
    papers_num = len(original)
    original = [paper for paper in original if paper.id not in continue_paper_ids]

    if not original:
        logger.warning(
            "No remaining items to process. They're all on the intermediate results."
        )
        return []
    else:
        logger.info("Skipping %d items.", papers_num - len(original))

    return original
