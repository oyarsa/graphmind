"""Prompts for extracting and annotating scientific terms from abstracts."""

from paper.gpt.prompts import PromptTemplate

MULTI = PromptTemplate(
    name="multi",
    template=f"""\
You are a scientific term extractor analyzing research papers. Extract \
terms as they appear in the abstract. Minimize modifications to the terms, unless it is \
to comply with the rules.

CRITICAL RULES:
1. Every term must be an exact substring of the abstract.
2. When multiple terms refer to the same entity, use a common term for all of them.
3. When a term contains an abbreviation, expand the abbreviation.
4. For each method identified, determine what tasks it helps accomplish.
5. For each metric identified, determine what tasks it evaluates.
6. For each resource identified, determine what tasks it supports.

Extract these categories:

1. TASKS: Core problems/objectives
  - Include both explicit and implicit tasks
  - Example: "cross-entropy training of deep neural networks"

2. METHODS: Technical approaches
  - Include algorithms, frameworks, and specific techniques
  - For each method, identify the tasks it addresses
  - Example: If abstract has "Hessian-free optimization for training neural networks"
    - Extract method: "Hessian-free optimization"
    - Create relation: ("Hessian-free optimization", "training neural networks")

3. METRICS: Evaluation measures
  - Include all quantitative measures
  - For each metric, identify the tasks it evaluates
  - Example: If abstract mentions "accuracy for machine translation"
    - Extract metric: "accuracy"
    - Create relation: ("accuracy", "machine translation")

4. RESOURCES: Datasets and tools
  - Include datasets, code, benchmarks
  - For each resource, identify the task it supports
  - Example: If abstract mentions "using WordNet for word sense disambiguation"
    - Extract resource: "WordNet"
    - Create relation: ("WordNet", "word sense disambiguation")

5. RELATIONS: Term relationships
  - Create "used for" relations between:
    - Each method and its corresponding tasks
    - Each metric and the tasks it evaluates
    - Each resource and the tasks it supports
  - Both terms must be exact copies from the abstract
  - Format: (source_term, target_task)

VERIFICATION STEPS:
1. For each method: Is there at least one "used for" relation to a task?
2. For each metric: Is there at least one "used for" relation to a task?
3. For each resource: Is there at least one "used for" relation to a task?
4. For each term: Does it contain an abbreviation that you can expand?
5. For each combination of terms: Do different words refer to the same entity? Can they \
be merged?

#####
-Data-
Abstract: {{abstract}}

#####
Output:
""",
)

TERM_USER_PROMPTS = {
    "multi": MULTI,
}
