"""Prompts for evaluating reviews to extract novelty ratings."""

from paper.gpt.prompts import PromptTemplate

SIMPLE = PromptTemplate(
    name="simple",
    template="""\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and a peer review evalauting the paper for a publication \
at a conference.

Based on this content, determine what novelty rating this reviewer gave to the paper. \
It should be a number from 1 to 5. This rating should reflect the following instructions:

How original is the approach? Does this paper break new ground in topic, methodology, or
content? How exciting and innovative is the research it describes?
Note that a paper could score high for originality even if the results do not show a convincing
benefit.
5 = Surprising: Significant new problem, technique, methodology, or insight -- no prior research
has attempted something similar.
4 = Creative: An intriguing problem, technique, or approach that is substantially different from
previous research.
3 = Respectable: A nice research contribution that represents a notable extension of prior
approaches or methodologies.
2 = Pedestrian: Obvious, or a minor improvement on familiar techniques.
1 = Significant portions have actually been done before or done better.

Based on this content, assign the paper a novelty rating from 1 to 5. First, generate
a rationale explaining why you gave the rating, then predict the novelty rating.

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Review:
{{review}}

#####
Output:
""",
)

TERNARY = PromptTemplate(
    name="ternary",
    template="""\
The following data contains information about a scientific paper. It includes the \
paper's title, abstract, and a peer review evalauting the paper for a publication \
at a conference.

Based on this content, determine what novelty rating this reviewer gave to the paper. \
It should be a number from 1 to 3. This rating should reflect the following instructions:

How original is the approach? Does this paper break new ground in topic, methodology, or
content? How exciting and innovative is the research it describes?
Note that a paper could score high for originality even if the results do not show a convincing
benefit.
3 = Positive: An intriguing problem, technique, or approach that is substantially different from
previous research.
2 = Neutral: A nice research contribution that represents a notable extension of prior
approaches or methodologies.
1 = Negative: Obvious, or a minor improvement on familiar techniques.

Based on this content, assign the paper a novelty rating from 1 to 5. First, generate
a rationale explaining why you gave the rating, then predict the novelty rating.

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Review:
{{review}}

#####
Output:
""",
)

REVIEW_CLASSIFY_USER_PROMPTS = {
    "simple": SIMPLE,
    "ternary": TERNARY,
}
