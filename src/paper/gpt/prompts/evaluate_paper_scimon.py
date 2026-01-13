"""Prompts for evaluating paper novelty using SciMON graph-extracted terms."""

from paper.gpt.prompts import PromptTemplate
from paper.gpt.prompts._shared import EVAL_SCALE

SIMPLE = PromptTemplate(
    name="simple",
    type_name="GPTFull",
    system="""\
Given inspiration sentences from related papers, give a novelty rating to a paper \
submitted to a high-quality scientific conference on a 1-5 scale.""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
paper's title and abstract, and some inspiration sentences from related papers. \
These sentences are meant to aid you in understanding whether the ideas in the paper \
are novel.

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Associated terms:
{{terms}}

#####
Output:
""",
)

SCIMON_CLASSIFY_USER_PROMPTS = {
    "simple": SIMPLE,
}
