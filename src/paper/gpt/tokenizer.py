"""Token counting and text truncation utilities."""

from __future__ import annotations

import tiktoken

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
