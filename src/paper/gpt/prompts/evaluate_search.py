"""Prompts for evaluating papers using web search for related work."""

from paper.gpt.prompts import PromptTemplate
from paper.gpt.prompts._shared import NOVELTY_5

# Shared instructions for search-based evaluation
_SEARCH_INSTRUCTIONS = """\
First, search the web for publications related to the paper. Your goal is to find relevant \
papers to compare the main paper with. This is important to determine if the paper's \
contributions are novel.

Based on this content, evaluate the paper's novelty. This should reflect how much the \
paper brings and develops new ideas previously unseen in the literature."""

_RATING_INSTRUCTIONS = f"""\
Rate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

Be thorough. When in doubt, tend towards lower ratings.

First generate the rationale for your novelty rating, then give the final rating (1-5)."""

_OUTPUT_FORMAT = """\
The output should have the following format:

```
Rationale: <text>

Label: <1-5>
```"""

SIMPLE = PromptTemplate(
    name="simple",
    system="""\
Given the following target paper, search the internet for relevant publications, and based \
on them, give a novelty rating from 1 to 5.
""",
    template=f"""\
The following data contains information about a scientific paper. It includes the \
paper's title and abstract.

{_SEARCH_INSTRUCTIONS}

{_RATING_INSTRUCTIONS}

{_OUTPUT_FORMAT}

#####
-Data-
Title: {{title}}
Abstract: {{abstract}}

#####
Output:
""",
)

ATTRIBUTION = PromptTemplate(
    name="attribution",
    system="""\
Given the following target paper, search the internet for relevant publications, and based \
on them, give a novelty rating from 1 to 5.
""",
    template=f"""\
The following data contains information about a scientific paper. It includes the \
paper's title and abstract.

{_SEARCH_INSTRUCTIONS}

{_RATING_INSTRUCTIONS}

The rationale must include the documents retrieved by web search. It must be pure plain \
text without any formatting. Instead of writing the titles and links to the documents \
inside the rationale, assign each a number and list them (number, title and link) at the \
bottom of the text.

{_OUTPUT_FORMAT}

#####
-Data-
Title: {{title}}
Abstract: {{abstract}}

#####
Output:
""",
)

SEARCH_EVAL_PROMPTS = {
    "simple": SIMPLE,
    "attribution": ATTRIBUTION,
}
