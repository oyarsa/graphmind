import logging
from dataclasses import dataclass
from typing import Any

import backoff
import openai
from openai import AsyncOpenAI
from openlimit import ChatRateLimiter
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("paper_hypergraph.gpt.run_gpt")

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
async def _call_gpt(rate_limiter: Any, client: AsyncOpenAI, chat_params: Any):
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

    # TODO: type this properly. OpenAI's types are a little convoluted.
    chat_params: Any = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=class_,
        seed=seed,
        temperature=temperature,
    )
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
