"""Demonstration prompts for abstract classification."""

from paper.gpt.prompts import PromptTemplate

SIMPLE = PromptTemplate(
    name="simple",
    template=f"""\
-Abstract:
{{abstract}}

-Background:
{{background}}

-Target:
{{target}}
""",
)

ABS_DEMO_PROMPTS = {
    "simple": SIMPLE,
}
