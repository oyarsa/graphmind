## Annotation Instructions: Rationale and Graph Component Evaluation

### Purpose

You are evaluating AI-generated rationales that assess the novelty of academic papers. Your task is to judge:

1. **How useful the rationale is** for understanding the paper's novelty (1-5 scale)
2. **Whether extracted graph components** (claims, methods, experiments) are accurate

### What You Will See

For each paper, you will be shown:

- Paper title and authors
- Link to the PDF
- Human review (for context)
- Model-generated rationale
- Extracted graph components: claims, methods, and experiments

### Task 1: Rate the Rationale (1-5)

Evaluate how useful the rationale would be for a reviewer trying to assess the paper's novelty. You are **not** judging whether it matches the human review—focus on whether it identifies relevant novelty aspects.

| Score | Label | Description |
|-------|-------|-------------|
| **5** | Very useful | Provides a solid foundation for evaluating novelty. Identifies key contributions and positions them well against prior work. A reviewer reading this would understand what's new and why it matters. |
| **4** | Useful | Captures most important novelty aspects with minor gaps or imprecisions. Still helpful for a reviewer. |
| **3** | Somewhat useful | Identifies some relevant points but misses important aspects or includes irrelevant discussion. Partial help to a reviewer. |
| **2** | Marginally useful | Touches on novelty superficially or focuses on wrong aspects. Limited value for assessment. |
| **1** | Not useful | Misses the point entirely, provides generic statements, or discusses aspects unrelated to novelty. |

**Note**: A rationale can be useful even if you disagree with its conclusions. Focus on whether the reasoning helps understand novelty, not whether you agree with the final assessment.

### Task 2: Evaluate Graph Components

For each extracted claim, method, and experiment, mark it as **Accurate** or **Inaccurate**.

A component is **Accurate** if:

- It is actually present in the paper
- It is correctly described (no misrepresentation)

A component is **Inaccurate** if:

- It does not appear in the paper
- It is partially correct but contains errors
- It misrepresents what the paper says

**Partial accuracy counts as inaccurate.** If a component is mostly right but has factual errors, mark it as inaccurate.

### Practical Guidelines

- **You do not need to read the entire paper.** Skim the abstract, introduction, method, and results sections to verify the rationale and components.
- **Use the PDF link** to check specific claims when needed.
- **Comments are optional** but helpful if you want to note something unusual.
- **When uncertain**, err on the side of marking components as inaccurate—we want to measure precision, not recall.
