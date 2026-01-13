"""Prompts for pairwise comparison of rationales."""

from paper.gpt.prompts import PromptTemplate

STANDARD = PromptTemplate(
    name="standard",
    type_name="pairwise_comparison",
    system="""\
You are evaluating and comparing two different rationales (labeled A and B) for the same scientific paper.
Your job is to determine which rationale is better according to a specific evaluation metric.
""",
    template="""\
# Paper Information
Title: {{title}}
Abstract: {{abstract}}

# Rationales to Compare
## Rationale A
{{rationale_a}}

## Rationale B
{{rationale_b}}

# Evaluation Instructions
Compare these two rationales and determine which one is better specifically in terms of "{{metric}}".

{{metric}}: {{definition}}

# Output Format
Your output must be structured as follows:
- Winner: A or B
- Explanation: A brief explanation of your decision

# Output
""",
)

PAIRWISE_COMPARISON_PROMPTS = {
    "standard": STANDARD,
}
