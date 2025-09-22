"""User prompt for the GPT models in TOML format."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

from paper.gpt.model import Prompt
from paper.util import read_resource


@dataclass(frozen=True, kw_only=True)
class PromptTemplate:
    """Prompt loaded from file with its name, template text, system prompt and type.

    The type refers to the Structured Output class that will be used by GPT to generate
    the result.

    The system prompt and type are optional.
    """

    name: str
    template: str
    system: str
    type_name: str

    def with_user(self, user_prompt: str) -> Prompt:
        """Create a Prompt instance with the given user prompt text."""
        return Prompt(system=self.system, user=user_prompt)


def load_prompts(name: str) -> Mapping[str, PromptTemplate]:
    """Load prompts from a TOML file in the prompts package.

    Args:
        name: Name of the TOML file in `paper.gpt.prompts`, without extension.

    Returns:
        Dictionary mapping prompt names to their text content.
    """
    text = read_resource("gpt.prompts", f"{name}.toml")
    return {
        p["name"]: PromptTemplate(
            name=p["name"],
            template=p["prompt"],
            system=p.get("system", ""),
            type_name=p.get("type", ""),
        )
        for p in tomllib.loads(text)["prompts"]
    }


def print_prompts(
    title: str, prompts: Mapping[str, PromptTemplate], *, detail: bool
) -> None:
    """Print the prompt names and types. Optionally, print the full template text."""
    if detail:
        print(">>>", title)
    else:
        print(title)

    for prompt in prompts.values():
        type_name = f"({prompt.type_name})" if prompt.type_name else ""
        if detail:
            sep = "-" * 80
            system = prompt.system or "default"
            lines = [
                sep,
                f"{prompt.name} {type_name}",
                sep,
                f"System: {system}",
                sep,
                prompt.template,
            ]
            print("\n".join(lines))
        else:
            print(f"- {prompt.name} {type_name}")
    print()
