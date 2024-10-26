"""User prompt for the GPT models in TOML format."""

import tomllib
from dataclasses import dataclass

from paper.util import read_resource


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    type_name: str
    template: str


def load_prompts(name: str) -> dict[str, PromptTemplate]:
    """Load prompts from a TOML file in the prompts package.

    Args:
        name: Name of the TOML file in `paper.gpt.prompts`, without extension.

    Returns:
        Dictionary mapping prompt names to their text content.
    """
    text = read_resource("gpt.prompts", f"{name}.toml")
    return {
        p["name"]: PromptTemplate(p["name"], p["type"], p["prompt"])
        for p in tomllib.loads(text)["prompts"]
    }
