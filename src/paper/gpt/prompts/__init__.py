"""User prompt for the GPT models in TOML format."""

import tomllib
from dataclasses import dataclass

from paper.util import read_resource


@dataclass(frozen=True, kw_only=True)
class PromptTemplate:
    """Prompt loaded from file with its name, template text and optional system prompt."""

    name: str
    template: str
    system: str | None


def load_prompts(name: str) -> dict[str, PromptTemplate]:
    """Load prompts from a TOML file in the prompts package.

    Args:
        name: Name of the TOML file in `paper.gpt.prompts`, without extension.

    Returns:
        Dictionary mapping prompt names to their text content.
    """
    text = read_resource("gpt.prompts", f"{name}.toml")
    return {
        p["name"]: PromptTemplate(
            name=p["name"], system=p.get("system"), template=p["prompt"]
        )
        for p in tomllib.loads(text)["prompts"]
    }


def print_prompts(
    title: str, prompts: dict[str, PromptTemplate], *, detail: bool
) -> None:
    """Print the prompt names and types. Optionally, print the full template text."""
    if detail:
        print(">>>", title)
    else:
        print(title)

    for prompt in prompts.values():
        if detail:
            sep = "-" * 80
            print(f"{sep}\n{prompt.name}\n{sep}\n{prompt.template}")
        else:
            print(f"- {prompt.name}")
    print()
