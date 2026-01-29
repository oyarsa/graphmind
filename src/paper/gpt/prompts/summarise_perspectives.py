"""Prompts for summarising novelty evaluations from multiple perspectives."""

from paper.gpt.prompts import PromptTemplate

# TODO: Rewrite this
SIMPLE = PromptTemplate(
    name="simple",
    type_name="PerspectiveSummary",
    system="""\
Given an evaluation of a paper's novelty from multiple perspectives, summarise the \
results and give a final judgement.
""",
    template=f"""\
The following data contains multiple evaluation of a given paper's novelty from different \
perspectives. Assess each perspective and combine them into a single rationale evaluating \
the paper as a whole. Provide a final label according to your rationale.

#####
-Data-
Perspectives:
{{perspectives}}

#####
""",
)

PERSPECTIVE_SUMMARY_PROMPTS = {
    "simple": SIMPLE,
}
