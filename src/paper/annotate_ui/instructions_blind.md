## Annotation Instructions: Rationale and Graph Component Evaluation (Blind)

### Purpose

You are evaluating AI-generated rationales that assess the novelty of academic papers. Your task is to judge:

1. **How useful the rationale is** for understanding the paper's novelty (1-5 scale)
2. **Whether extracted graph components** (claims, methods, experiments) are accurate

### What You Will See

For each paper, you will be shown:

- Paper title and authors
- Link to the PDF
- AI-generated rationale explaining the paper's novelty
- Extracted graph components: claims, methods, and experiments

### Task 1: Rate the Rationale (1-5)

Evaluate how well the rationale identifies and explains the paper's novelty aspects.

| Score | Label | Description |
|-------|-------|-------------|
| **5** | Very useful | Comprehensively identifies the key novelty aspects. Correctly distinguishes what is new from what builds on prior work. |
| **4** | Useful | Identifies most novelty aspects with minor omissions or imprecisions. |
| **3** | Somewhat useful | Identifies some novelty aspects but misses important ones, or includes irrelevant points. |
| **2** | Marginally useful | Mentions novelty superficially or focuses on wrong aspects. Limited value for assessment. |
| **1** | Not useful | Fails to identify novelty, or makes incorrect claims about what is novel. |

**Guidelines:**

- Skim the abstract and relevant sections (introduction, contributions, related work) to understand the paper's actual novelty
- Focus on whether the rationale correctly identifies *what* is novel about the paper
- A good rationale should be specific (e.g., "novel loss function for X" not just "novel approach")

### Task 2: Evaluate Graph Components

For each extracted claim, method, and experiment, mark it as **Accurate** or **Inaccurate**.

A component is **Accurate** if:

- It is actually present in the paper
- It is correctly described (semantic equivalence is acceptable)

A component is **Inaccurate** if:

- It does not appear in the paper
- It misrepresents what the paper says

**Examples:**

- Paper says "We use BERT" → Extraction says "transformer-based language model" → **Accurate** (semantically equivalent)
- Paper says "We evaluate on GLUE" → Extraction says "evaluated on SuperGLUE" → **Inaccurate** (wrong benchmark)
- Extraction mentions a method not discussed in the paper → **Inaccurate**

### Practical Guidelines

- **You do not need to read the entire paper.** Skim the abstract, introduction, method, and results sections to verify the rationale and components.
- **Use the PDF link** to check specific claims when needed.
- **Comments are optional** but helpful if you want to note something unusual.
- **When uncertain**, err on the side of marking components as inaccurate—we want to measure precision, not recall.
