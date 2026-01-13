"""Demonstration prompts for novelty evaluation."""

from paper.gpt.prompts import PromptTemplate

ABSTRACT = PromptTemplate(
    name="abstract",
    template="""\
Title: {{title}}
Abstract: {{abstract}}
Novelty rating: {{rating}}
Rationale: {{rationale}}
""",
)

MAINTEXT = PromptTemplate(
    name="maintext",
    template="""\
Title: {{title}}
Abstract: {{abstract}}

Main text:
{{main_text}}

Novelty rating: {{rating}}
Rationale: {{rationale}}
""",
)

EVALUATE_DEMONSTRATION_PROMPTS = {
    "abstract": ABSTRACT,
    "maintext": MAINTEXT,
}
