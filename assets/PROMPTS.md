# Prompts Referenced in the Paper

This document is a companion reference for readers of the paper. Each section
corresponds to a prompted method, preprocessing step, baseline, or evaluation
procedure described in the text. Placeholders in `{curly braces}` are filled at
runtime.

## Shared Novelty Rubric

Novelty-scoring prompts use the following 1-5 scale.

```
1 = Not novel: Significant portions done before or done better
2 = Minor improvement on familiar techniques
3 = Notable extension of prior approaches
4 = Substantially different from previous research
5 = Significant new problem, technique, or insight
```

---

## Hierarchical Graph Construction

Used in the method section where the paper describes constructing the
hierarchical graph from the target paper's title, abstract, and full text.

```
Given the following scientific paper, extract the important entities from the text and the relationships between them. The goal is to build a faithful and concise representation of the paper that captures the most important elements necessary to evaluate its novelty.
```

```
The following data contains information about a scientific paper. It includes the
paper's title, abstract, and the main text. The goal is to represent all the relevant
information from the paper as a graph.

Your task is to extract entities of the following types and the relationships between
them. All entities must have different text descriptions.
- `title`: the title of the paper.
- `primary_area`: what scientific primary area the paper is from. It must be one from
the following list: {primary_areas}.
- `keywords`: words that summarises what are the key aspects of the paper.
- `tldr`: a sentence that summarises the paper from the abstract.
- `claim`: summarise what the paper claims to contribute, especially claims made in the
abstract, introduction, discussion and conclusion. Pay attention to the key phrases
that highlight new findings or interpretations.
- `method`: for each `claim`, identify the methods used to validate the claims from the
method sections. These include the key components: algorithms, theoretical framework
or novel techniques introduced.
- `experiment`: what models, baselines, datasets, etc. were used in experiments to
validate the `methods`.

Extract these entities and the relationships between them. The paper `title` is the root
entity. You must follow these rules when generating the entities and relationships:

- The `title` connects to the `primary_area`, the `keywords` and the `tldr` sentence.
- The `tldr` sentence connects to all the claims.
- Each `claim` must connect to one or more `methods`.
- Each `method` must connect to one or more `experiments`.
- Every `method` and `experiment` must connected to at least one entity.
- Each `method` has a unique index starting from 0. These are indices from the `methods`
  list in the graph. `claims` will connect to methods by these indices.
- Each `experiment` has a unique index starting from 0. These are indices from the
  `experiments` list in the graph. `methods` will connect to experiments by these
  indices.

The `claim`, `method` and `experiments` entities also contain the following fields:
- detail: Detailed description of the entity.
- excerpts: Collection of text excerpts about the entity copied verbatim from the paper.

All entity types should be present in the output. None of the lists in the output can be
empty.

#####
-Data-
Title: {title}
Abstract: {abstract}

Main text:
{main_text}

#####
Output:
```

---

## Citation Polarity Classification

Used in the related-paper processing step to decide whether each citation
context is supporting or contrasting.

```
You are given a citation context from a scientific paper that mentions. Your task is to determine the polarity of the citation context as 'positive' or 'negative'.

The polarity represents whether the citation context is supporting the paper's goals ('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Citation context: {context}
#####
Output:
```

---

## Abstract Splitting for Semantic Neighbors

Used in semantic-neighbor categorisation to split abstracts into background and
methodology/target components before computing similarities.

```
Given the paper abstract, your goal is to extract the paper's background context and
target.

The background context describes the problem setup, motivation, rationale, task and previous knowledge.
The target describes the methods, objectives, goals, findings, results or implications.

The output will contain two fields: `background` and `target`. For each of them, collect the sentences of the appropriate type. The output should be the relevant sentences combined for each type. All sentences in the abstract must be either a background context or a target sentence. No sentence in the abstract should be missing from the output.

#####
{demonstrations}
-Data-

Abstract: {abstract}

#####
Output:
```

---

## Supporting Related-Paper Summary

Used when the selected related paper is treated as supporting evidence for the
target paper.

```
Compare the following two papers. The second paper has a supportive relationship to the first - it contains findings that strengthen or corroborate the first paper's claims.

Summarise how the second paper supports the first. Be direct and concise (2-3 sentences). Focus on the specific supporting evidence or methodological similarities.

IMPORTANT: Start your summary with the actual finding or methodology, not with phrases like "The paper", "This paper", "The supporting paper", or "The second paper". Jump straight into what the connection is.

#####
-Data-
# Paper being evaluated

**Title**: {title_main}
**Abstract**: {abstract_main}

# Supporting paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Summary:
```

---

## Contrasting Related-Paper Summary

Used when the selected related paper is treated as contrasting evidence for the
target paper.

```
Compare the following two papers. The second paper has a contrasting relationship to the first - it presents different approaches or findings that contrast with the first paper's claims.

Summarise how the second paper contrasts with the first. Be direct and concise (2-3 sentences). Focus on the specific differences in approach, methodology, or findings.

IMPORTANT: Start your summary with the actual finding or methodology, not with phrases like "The paper", "This paper", "The contrasting paper", or "The second paper". Jump straight into what the difference is.

#####
-Data-
# Paper being evaluated

**Title**: {title_main}
**Abstract**: {abstract_main}

# Contrasting paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Summary:
```

---

## GraphMind Novelty Assessment

Used as the final GraphMind scoring prompt after constructing the hierarchical
graph and summarising supporting and contrasting related papers.

```
Given the following target paper, a summary and a selection of related papers separated by whether they're supporting or contrasting the main paper, provide a structured novelty evaluation on a 1-5 scale.
```

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

The paper summary describes the most important information about the paper and its contents. It summarises key aspects, which you can use to build a more comprehensive understanding of the paper.

The related papers are split into "supporting" papers (those that corroborate the paper's ideas, methods, approach, etc.) and "contrasting" papers (those that go against the paper's ideas). Use these related papers to understand the context around the main paper, so you know what other works exist in comparison with the main paper.

Each related paper is presented with its ID, title, and relevant information. When referencing evidence from these papers, you MUST include the paper's ID and title.

Based on this, decide whether the paper is novel. It is novel if brings new ideas or develops new ideas previously unseen. Make sure that the ideas are truly unique. The paper is not novel if anything similar to it has been done before. Be very thorough. When in doubt, tend towards the not novel label.

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
{novelty_rubric}

IMPORTANT: When creating evidence items, structure them as:
- text: The evidence text describing the finding
- paper_id: The ID of the paper (if available)
- paper_title: The title of the paper
- source: Either "citations" or "semantic" to indicate how this paper was found

If the paper approval decision is "True", the novelty rating should be 4 or 5 (novel).

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}
Approval decision: {approval}

Paper summary:
{graph}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

---

## Basic Prompting Baseline

Used for the baseline that scores novelty from the title and abstract only.

```
Given a paper's title and abstract, provide a structured novelty evaluation. This evaluation uses only the paper's own content without any related papers or graph summary.
```

```
The following data contains information about a scientific paper. It includes only the paper's title and abstract.

Based solely on the title and abstract, evaluate the paper's novelty. Consider:
- What problem or question is being addressed?
- What methods or approaches are proposed?
- How significant do the contributions appear to be?

You do not have access to related papers, so base your assessment purely on how the paper presents its contributions and how novel the described approach sounds.

Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. What aspects suggest the paper might be novel (leave evidence lists empty since no related papers are available)
3. What aspects suggest the paper might not be novel (leave evidence lists empty)
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{novelty_rubric}

Be conservative in your ratings since you don't have related work context.

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

#####
```

---

## Search-Augmented Prompting Baseline

Used for the baseline that asks the model to search for related publications
before scoring novelty.

```
Given the following target paper, search the internet for relevant publications, and based on them, give a novelty rating from 1 to 5.
```

````
The following data contains information about a scientific paper. It includes the paper's title and abstract.

First, search the web for publications related to the paper. Your goal is to find relevant papers to compare the main paper with. This is important to determine if the paper's contributions are novel.

Based on this content, evaluate the paper's novelty. This should reflect how much the paper brings and develops new ideas previously unseen in the literature.

Rate the paper's novelty on a 1-5 scale:
{novelty_rubric}

Be thorough. When in doubt, tend towards lower ratings.

First generate the rationale for your novelty rating, then give the final rating (1-5).

The rationale must include the documents retrieved by web search. It must be pure plain text without any formatting. Instead of writing the titles and links to the documents inside the rationale, assign each a number and list them (number, title and link) at the bottom of the text.

The output should have the following format:

```
Rationale: <text>

Label: <1-5>
```

#####
-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
````

---

## SciMON Baseline

Used for the SciMON baseline, where inspiration sentences from related papers
are supplied to the novelty-scoring model.

```
Given inspiration sentences from related papers, give a novelty rating to a paper submitted to a high-quality scientific conference on a 1-5 scale.
```

```
The following data contains information about a scientific paper. It includes the paper's title and abstract, and some inspiration sentences from related papers. These sentences are meant to aid you in understanding whether the ideas in the paper are novel.

Based on this, evaluate the paper's novelty on a 1-5 scale:
{novelty_rubric}

Be thorough. When in doubt, tend towards lower ratings.

First, generate the rationale for your novelty rating, then give the final rating (1-5).

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Associated terms:
{terms}

#####
Output:
```

---

## NovAScore Baseline

Used for the NovAScore baseline to extract Atomic Content Units from paper
abstracts.

```
The following data contains information about a scientific paper. It includes the paper's title and abstract. Based on this, your task is to extract all atomic content units from the paper.

1. First, extract the list of all atomic content units (ACUs) from a given document. An ACU is an elementary information unit in the document that does not require further division. When identifying any named entity, temporal entity, location entity, or attribute, avoid using indirect references. Instead, specify the actual entity, attribute, or noun directly. For example, replace 'this company' with the actual name of the company, 'this location' with the actual location name, 'it' with the actual subject being referred, etc.

2. Then, summarize the given document.

3. Finally, using the summary, identify the most salient ACUs from the full list of ACUs. The salient ACUs should be those explicitly mentioned in the summary.

#####
-Demonstration-
Example document:

Enzon Inc. said tests of a new anti-cancer drug to treat a form of leukemia showed positive results, Abraham Abuchowski, president of the pharmeutical company, told the annual meeting. Enzon also said it filed its first new drug application with the FDA for an orphan drug that treats severe combined immunodeficiency disease, or SCID. The company has about 10.8 million common shares outstanding. Eastman Kodak Co. ownas an 18% interest in Enzon.

Example ACUs:
- Enzon Inc. reported positive results from tests of a new anti-cancer drug for leukemia.
- Abraham Abuchowski is the president of Enzon Inc.
- Enzon Inc. filed a new drug with the FDA to treat immunodeficiency disease.
- Enzon Inc. has about 10.8 million common shares outstanding.

-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
```

---

## SC4ANM Baseline

Used for the SC4ANM baseline, where novelty is predicted from Introduction,
Results, and Discussion sections.

```
You are a scientific paper reviewer. Based on the following sections of an academic paper, predict its novelty score.
```

```
The following data contains key sections (Introduction, Results, Discussion) of a scientific paper. These are the sections most relevant to assessing the paper's novelty.

Based solely on the paper's section content, evaluate its novelty. You do not have access to related papers, so base your assessment purely on how the paper presents its contributions.

Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. What aspects suggest the paper might be novel (leave evidence lists empty since no related papers are available)
3. What aspects suggest the paper might not be novel (leave evidence lists empty)
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{novelty_rubric}

Rate the paper fairly based on its actual contributions. Use the full 1-5 scale:
- Rate 1-2 only if the paper clearly describes incremental or derivative work
- Rate 4-5 if the paper describes genuinely new ideas, methods, or insights
- Rate 3 for papers with moderate novelty or when uncertain

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

{ird_sections}

#####
```

---

## RAG-Novelty Baseline

Used for the RAG-Novelty baseline, where retrieved related papers and
publication-year context are supplied to the novelty-scoring model.

```
You are a scientific paper reviewer assessing the novelty of a research paper. You have been provided with several related papers retrieved from the literature to help inform your assessment.
```

```
The following data contains a target paper and related papers retrieved by similarity to the target paper's abstract. Use the related papers to inform your novelty assessment.

Evaluate the paper's novelty based on its contributions compared to the related work.

IMPORTANT: The existence of related papers does NOT mean the paper lacks novelty. Related papers provide context, but novelty depends on whether this paper makes meaningful new contributions beyond what exists. A paper can be highly novel even if similar topics have been explored, as long as it offers significant new insights, methods, or perspectives.

Consider:
- Does the paper introduce new ideas, methods, or perspectives not present in related work?
- Does it solve problems in a substantially different or better way?
- Does it open new research directions or provide significant new insights?

Rate the paper fairly based on its actual contributions:
{novelty_rubric}

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
{novelty_rubric}

IMPORTANT: When creating evidence items, structure them as:
- text: The evidence text describing the finding
- paper_id: The ID of the paper (if available)
- paper_title: The title of the paper
- source: Either "citations" or "semantic" to indicate how this paper was found

After forming your initial assessment, reflect on whether you may be over- or under-estimating novelty given the evidence from the related papers.

{year_context}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Retrieved Related Papers:
{retrieved_papers}

#####
```

---

## Rationale Evaluation

Used for the GPT-4o pairwise tournament that compares generated rationales under
each evaluation metric.

```
You are evaluating and comparing two different rationales (labeled A and B) for the same scientific paper.
Your job is to determine which rationale is better according to a specific evaluation metric.
```

```
# Paper Information
Title: {title}
Abstract: {abstract}

# Rationales to Compare
## Rationale A
{rationale_a}

## Rationale B
{rationale_b}

# Evaluation Instructions
Compare these two rationales and determine which one is better specifically in terms of "{metric}".

{metric}: {definition}

# Output Format
Your output must be structured as follows:
- Winner: A or B
- Explanation: A brief explanation of your decision

# Output
```
