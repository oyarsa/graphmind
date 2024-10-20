import logging
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("paper_hypergraph.gpt.run_gpt")

# Cost in $ per 1M tokens: (input cost, output cost)
# From https://openai.com/api/pricing/
MODEL_COSTS = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4o-2024-08-06": (2.5, 10),
}
MODELS_ALLOWED = sorted(MODEL_COSTS)


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate API request based on the model and input/output tokens.

    NB: prompt_tokens/completion_tokens is the name given to input/output tokens in the
    usage object from the OpenAI result.

    Args:
        prompt_tokens: the input tokens for the API
        completion_tokens: the output tokens from the API

    Returns:
        The total cost of the request. If the model is invalid, returns NaN.
    """
    if model not in MODEL_COSTS:
        return float("nan")

    input_cost, output_cost = MODEL_COSTS[model]
    return prompt_tokens / 1e6 * input_cost + completion_tokens / 1e6 * output_cost


@dataclass(frozen=True)
class GPTResult[T]:
    """Result of a GPT request and its full API cost."""

    result: T
    cost: float


def run_gpt[T: BaseModel](
    class_: type[T],
    client: OpenAI,
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
    """
    if model not in MODELS_ALLOWED:
        raise ValueError(
            f"Invalid model: {model!r}. Should be one of: {MODELS_ALLOWED}."
        )

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=class_,
            seed=seed,
            temperature=temperature,
        )
    except Exception:
        logger.exception("Error making API request")
        return GPTResult(result=None, cost=float("nan"))

    usage = completion.usage
    if usage is not None:
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)
    else:
        cost = 0

    parsed = completion.choices[0].message.parsed

    return GPTResult(result=parsed, cost=cost)
