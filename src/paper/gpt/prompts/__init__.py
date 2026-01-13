"""User prompts for the GPT models.

Prompts are defined in Python modules within this package. Each module exports
a dictionary mapping prompt names to PromptTemplate instances.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from paper.gpt.model import Prompt

__all__ = ["PromptTemplate", "print_prompts"]


@dataclass(frozen=True, kw_only=True)
class PromptTemplate:
    """Prompt loaded from file with its name, template text, system prompt and type.

    The type refers to the Structured Output class that will be used by GPT to generate
    the result.

    The system prompt and type are optional.
    """

    name: str
    template: str
    system: str = ""
    type_name: str = ""

    def with_user(self, user_prompt: str) -> Prompt:
        """Create a Prompt instance with the given user prompt text."""
        return Prompt(system=self.system, user=user_prompt)

    def format(self, **kwargs: str) -> str:
        """Format the template with the given keyword arguments."""
        return self.template.format(**kwargs)


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
