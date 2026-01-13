"""Prompts for classifying citation context polarity."""

from paper.gpt.prompts import PromptTemplate

SENTENCE = PromptTemplate(
    name="sentence",
    template="""\
You are given a citation context from a scientific paper that mentions. Your task is to \
determine the polarity of the citation context as 'positive' or 'negative'.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Citation context: {{context}}
#####
Output:
""",
)

SIMPLE = PromptTemplate(
    name="simple",
    template="""\
You are given a main paper and a reference with a citation context. Your task is to \
determine the polarity of the citation context as 'positive' or 'negative', given the \
main paper's title, the reference's title, and the citation context where the main \
paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {{main_title}}
Reference title: {{reference_title}}
Citation context: {{context}}
#####
Output:
""",
)

FULL = PromptTemplate(
    name="full",
    template="""\
You are given a main paper and a reference with a citation context. Your task is to
determine the polarity of the citation context as 'positive' or 'negative', given the
main paper's title and abstract, the reference's title and abstract, and the citation
context where the main paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {{main_title}}
Main paper abstract: {{main_abstract}}

Reference title: {{reference_title}}
Reference abstract: {{reference_abstract}}

Citation context: {{context}}
#####
Output:
""",
)

CONTEXT_USER_PROMPTS = {
    "sentence": SENTENCE,
    "simple": SIMPLE,
    "full": FULL,
}
