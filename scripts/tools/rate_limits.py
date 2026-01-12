"""Check OpenAI API rate limits for different models."""

import asyncio
import os
import sys
from dataclasses import dataclass

import aiohttp
import typer


@dataclass(frozen=True, kw_only=True)
class RateLimit:
    """Rate limit information from OpenAI API response headers."""

    requests: int
    tokens: int


async def fetch_rate_limits(
    session: aiohttp.ClientSession,
    model: str,
    api_key: str,
) -> RateLimit | None:
    """Fetch rate limits for a specific model from the OpenAI API. Returns None on error."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        ) as response:
            # Even if the model is invalid, we should still get rate limit headers
            requests = response.headers.get("x-ratelimit-limit-requests")
            tokens = response.headers.get("x-ratelimit-limit-tokens")

            if not requests or not tokens:
                return None

            return RateLimit(requests=int(requests), tokens=int(tokens))
    except (aiohttp.ClientError, ValueError):
        return None


async def _main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable not set")

    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1-preview",
        "o1-mini",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_rate_limits(session, model, api_key) for model in models]
        results = await asyncio.gather(*tasks)

        for model, limits in zip(models, results):
            print(f">>> {model}")
            if limits:
                print(f"x-ratelimit-limit-requests {limits.requests:,}")
                print(f"x-ratelimit-limit-tokens {limits.tokens:,}")
            else:
                print("Failed to fetch rate limits")
            print()


def main() -> None:
    """Check rate limits (requests and tokens per minute) for multiple OpenAI models."""
    asyncio.run(_main())


if __name__ == "__main__":
    typer.run(main)
