"""Shared prompt fragments used across multiple prompt templates."""

# Rating scale
NOVELTY_5 = """\
1 = Not novel: Significant portions done before or done better
2 = Minor improvement on familiar techniques
3 = Notable extension of prior approaches
4 = Substantially different from previous research
5 = Significant new problem, technique, or insight"""

# Instructions
CONSERVATIVE = "Be thorough. When in doubt, tend towards lower ratings."

RATIONALE_BASIC = """\
First, generate the rationale for your novelty rating, then give the final rating (1-5)."""

RATIONALE_STRUCTURED = f"""\
Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. Specific evidence from related papers that supports the paper's novelty
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that supports novelty
   - Indicate whether this is a citation-based or semantic-based related paper
3. Specific evidence from related papers that contradicts the paper's novelty
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that contradicts novelty
   - Indicate whether this is a citation-based or semantic-based related paper
4. Key technical comparisons that influenced your decision
5. Your final assessment and conclusion about the paper's novelty
6. A novelty rating from 1 to 5:
{NOVELTY_5}

IMPORTANT: When creating evidence items, structure them as:
- text: The evidence text describing the finding
- paper_id: The ID of the paper (if available)
- paper_title: The title of the paper
- source: Either "citations" or "semantic" to indicate how this paper was found"""

RATIONALE_NO_RELATED = f"""\
Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. What aspects suggest the paper might be novel (leave evidence lists empty since no \
related papers are available)
3. What aspects suggest the paper might not be novel (leave evidence lists empty)
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{NOVELTY_5}

Be conservative in your ratings since you don't have related work context."""

EVAL_SCALE = f"""\
Based on this, evaluate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

{CONSERVATIVE}

{RATIONALE_BASIC}"""

EVAL_SCALE_STRUCTURED = f"""\
Based on this, decide whether the paper is novel. It is novel if brings new ideas or \
develops new ideas previously unseen. Make sure that the ideas are truly unique. The \
paper is not novel if anything similar to it has been done before. Be very thorough. \
When in doubt, tend towards the not novel label.

{RATIONALE_STRUCTURED}"""

# Balanced version without conservative bias
EVAL_SCALE_BALANCED = f"""\
Evaluate the paper's novelty based on its contributions compared to the related work.

IMPORTANT: The existence of related papers does NOT mean the paper lacks novelty. \
Related papers provide context, but novelty depends on whether this paper makes \
meaningful new contributions beyond what exists. A paper can be highly novel even \
if similar topics have been explored, as long as it offers significant new insights, \
methods, or perspectives.

Consider:
- Does the paper introduce new ideas, methods, or perspectives not present in related work?
- Does it solve problems in a substantially different or better way?
- Does it open new research directions or provide significant new insights?

Rate the paper fairly based on its actual contributions:
{NOVELTY_5}

{RATIONALE_STRUCTURED}"""

# Context descriptions
GRAPH_INTRO = """\
The paper summary describes the most important information about the paper and its \
contents. It summarises key aspects, which you can use to build a more comprehensive \
understanding of the paper."""

RELATED_INTRO = """\
The related papers are split into "supporting" papers (those that corroborate the paper's \
ideas, methods, approach, etc.) and "contrasting" papers (those that go against the \
paper's ideas). Use these related papers to understand the context around the main paper, \
so you know what other works exist in comparison with the main paper."""

RELATED_WITH_IDS = """\
Each related paper is presented with its ID, title, and relevant information. When \
referencing evidence from these papers, you MUST include the paper's ID and title."""
