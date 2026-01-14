"""Prompts for evaluating paper novelty using graph summaries."""

from paper.gpt.prompts import PromptTemplate
from paper.gpt.prompts._shared import (
    EVAL_SCALE,
    EVAL_SCALE_BALANCED,
    EVAL_SCALE_STRUCTURED,
    GRAPH_INTRO,
    NOVELTY_5,
    RATIONALE_NO_RELATED,
    RATIONALE_STRUCTURED,
    RELATED_INTRO,
    RELATED_WITH_IDS,
)

FULL_GRAPH = PromptTemplate(
    name="full-graph",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
Output:
""",
)

ONLY_GRAPH = PromptTemplate(
    name="only-graph",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
Output:
""",
)

TITLE_GRAPH = PromptTemplate(
    name="title-graph",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
Output:
""",
)

RELATED = PromptTemplate(
    name="related",
    type_name="GPTStructured",
    system="""\
Evaluate a paper's novelty based on its abstract and related work.
""",
    template=f"""
You are evaluating a scientific paper's novelty based only on its abstract and related papers.

The related papers are split into:
- "Supporting" papers: work that builds on similar ideas or methods
- "Contrasting" papers: work that takes different approaches to similar problems

**Your task:** Determine if the related papers provide clear evidence that this paper's \
claimed contributions have already been done.

**Important context:** You only have access to abstracts, not full papers. Without full \
details, you should give the benefit of doubt to the paper's claimed contributions. Only \
conclude lack of novelty if the related papers explicitly describe doing the same thing.

**Evaluation steps:**
1. What does the abstract claim as its specific contributions?
2. Do any related papers explicitly claim to have done the same thing?
3. If unclear or ambiguous, favour the paper being novel (you lack information to prove otherwise)

Remember: Most papers in an active research area will have related work. The question is \
whether this paper's specific contribution has been done before, not whether similar research \
exists.

{RATIONALE_STRUCTURED}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
""",
)

SEMANTIC_ONLY = PromptTemplate(
    name="semantic-only",
    type_name="GPTStructured",
    system="""\
Given the following target paper, a summary and semantically similar papers, provide \
a structured novelty evaluation on a 1-5 scale.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and semantically similar papers found \
through search.

{GRAPH_INTRO}

The related papers were found through semantic similarity search. Note that semantic \
search may return papers that are only tangentially related or share surface-level \
similarities without being directly comparable work. Be cautious about using semantic \
matches as strong evidence against novelty - focus on whether the papers actually \
address the same core problem with the same approach.

{RELATED_WITH_IDS}

Evaluate the paper's novelty, but be aware that semantic similarity does not imply \
lack of novelty. A paper can be highly novel even if semantically similar papers exist, \
as long as its core contributions are meaningfully different.

{RATIONALE_STRUCTURED}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
""",
)

SANS = PromptTemplate(
    name="sans",
    type_name="GPTStructured",
    system="""\
Given a paper's title and abstract, provide a structured novelty evaluation. This \
evaluation uses only the paper's own content without any related papers or graph summary.
""",
    template=f"""
The following data contains information about a scientific paper. It includes only the \
paper's title and abstract.

Based solely on the title and abstract, evaluate the paper's novelty. Consider:
- What problem or question is being addressed?
- What methods or approaches are proposed?
- How significant do the contributions appear to be?

You do not have access to related papers, so base your assessment purely on how the \
paper presents its contributions and how novel the described approach sounds.

{RATIONALE_NO_RELATED}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

#####
""",
)

NOREL_GRAPH = PromptTemplate(
    name="norel-graph",
    type_name="GPTStructured",
    system="""\
Given the following target paper and its graph summary, provide a structured novelty \
evaluation based solely on the paper's own content.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, abstract, and a summary of its key points extracted as a graph.

{GRAPH_INTRO}

Note: This evaluation does NOT include related papers - only the main paper's content \
and graph summary are available.

Based solely on the paper's content and graph summary, evaluate how novel the described \
approach and contributions appear to be.

{RATIONALE_NO_RELATED}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Paper summary:
{{graph}}

#####
""",
)

FULL_GRAPH_POSITIVE = PromptTemplate(
    name="full-graph-positive",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers supporting \
the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

The related papers support the main paper, i.e. they corroborate the paper's ideas, methods, \
approach, etc. Use these related papers to understand the context around the main paper, \
so you know what other works exist in comparison with the main paper.

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

#####
Output:
""",
)

FULL_GRAPH_NEGATIVE = PromptTemplate(
    name="full-graph-negative",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers \
contrasting the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

The related papers contrast the main paper, i.e. they go against the main paper's \
ideas. Use these related papers to understand the context around the main paper, \
so you know what other works exist in comparison with the main paper.

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
Output:
""",
)

ATTRIBUTION_GRAPH = PromptTemplate(
    name="attribution-graph",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, give it a novelty rating.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}
#####

Based on this, evaluate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

Be very thorough. When in doubt, tend towards lower ratings.

For each related paper, assign a code to them. When discussing related papers in the \
rationale, be specific and use the paper's code. After the rationale, add a references \
section with the code and name for each paper. For brevity, don't use the paper names \
in the rationale, only the code. Make sure that all referenced papers appear in the \
references section.

First, generate the rationale for your novelty rating, then give the final rating (1-5).
""",
)

FULL_GRAPH_BASIC = PromptTemplate(
    name="full-graph-basic",
    type_name="GPTFull",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, give it a novelty rating \
on a 1-5 scale.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{EVAL_SCALE}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
Output:
""",
)

FULL_GRAPH_STRUCTURED = PromptTemplate(
    name="full-graph-structured",
    type_name="GPTStructured",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, provide a structured \
novelty evaluation on a 1-5 scale.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_STRUCTURED}

If the paper approval decision is "True", the novelty rating should be 4 or 5 (novel).

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}
Approval decision: {{approval}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
""",
)

FULL_GRAPH_BALANCED = PromptTemplate(
    name="full-graph-balanced",
    type_name="GPTStructured",
    system="""\
Given the following target paper, a summary and a selection of related papers separated \
by whether they're supporting or contrasting the main paper, provide a structured \
novelty evaluation.
""",
    template=f"""
The following data contains information about a scientific paper. It includes the \
main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_BALANCED}

#####
{{demonstrations}}

-Data-
Title: {{title}}
Abstract: {{abstract}}

Paper summary:
{{graph}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
""",
)

DEBUG_RANDOM = PromptTemplate(
    name="debug-random",
    type_name="GPTFull",
    system="""\
Give a novelty rating for the following paper on a 1-5 scale, and explain your decision.
""",
    template="""
The following data contains information about a scientific paper. It includes the \
paper's title and abstract.

Assign a random label between 1 and 5. Your rationale should be "debug".

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
""",
)

SIMPLE_BASIC = PromptTemplate(
    name="simple-basic",
    type_name="GPTStructured",
    system="""\
Given a paper title and abstract along with related papers found through semantic search, \
provide a structured novelty evaluation for this paper.
""",
    template=f"""
The following data contains information about a paper that needs novelty assessment. It includes \
the paper's title and abstract, along with related papers discovered through semantic search.

The related papers are ordered by relevance and include both supporting papers (those with \
similar research directions or methods) and contrasting papers (those with different approaches \
to similar problems). Use these related papers to understand the research landscape and assess \
whether the target paper presents novel contributions.

{RELATED_WITH_IDS}

Based on this information, evaluate the paper's novelty on a 1-5 scale. Be thorough in your \
assessment and when in doubt, tend towards lower ratings.

Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach based on the title and abstract
2. Specific evidence from related papers that supports the paper's novelty
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that supports novelty
   - Indicate that this is a search-based related paper
3. Specific evidence from related papers that contradicts the paper's novelty
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that contradicts novelty
   - Indicate that this is a search-based related paper
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{NOVELTY_5}

IMPORTANT: When creating evidence items, structure them as:
- text: The evidence text describing the finding
- paper_id: The ID of the paper (if available)
- paper_title: The title of the paper
- source: "search" to indicate this paper was found through semantic search

#####
{{demonstrations}}

-Data-
Title: {{title}}

Abstract: {{abstract}}

Supporting papers:
{{positive}}

Contrasting papers:
{{negative}}

#####
""",
)

GRAPH_EVAL_USER_PROMPTS = {
    "full-graph": FULL_GRAPH,
    "only-graph": ONLY_GRAPH,
    "title-graph": TITLE_GRAPH,
    "related": RELATED,
    "semantic-only": SEMANTIC_ONLY,
    "sans": SANS,
    "norel-graph": NOREL_GRAPH,
    "full-graph-positive": FULL_GRAPH_POSITIVE,
    "full-graph-negative": FULL_GRAPH_NEGATIVE,
    "attribution-graph": ATTRIBUTION_GRAPH,
    "full-graph-basic": FULL_GRAPH_BASIC,
    "full-graph-structured": FULL_GRAPH_STRUCTURED,
    "full-graph-balanced": FULL_GRAPH_BALANCED,
    "debug-random": DEBUG_RANDOM,
    "simple-basic": SIMPLE_BASIC,
}
