# Used Prompts

Reference dump of every prompt template invoked by the pipeline, organised by
task. Each entry shows the variant name, system prompt, and template body.
Placeholders in `{curly braces}` are filled at runtime.

Source modules live in `src/paper/gpt/prompts/`. Shared fragments referenced
below are defined in `_shared.py`.

## Shared fragments

### `NOVELTY_5` — 1-5 novelty rating scale

```
1 = Not novel: Significant portions done before or done better
2 = Minor improvement on familiar techniques
3 = Notable extension of prior approaches
4 = Substantially different from previous research
5 = Significant new problem, technique, or insight
```

### `CONSERVATIVE`

```
Be thorough. When in doubt, tend towards lower ratings.
```

### `RATIONALE_BASIC`

```
First, generate the rationale for your novelty rating, then give the final rating (1-5).
```

### `RATIONALE_STRUCTURED`

```
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
- source: Either "citations" or "semantic" to indicate how this paper was found
```

### `RATIONALE_NO_RELATED`

```
Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. What aspects suggest the paper might be novel (leave evidence lists empty since no related papers are available)
3. What aspects suggest the paper might not be novel (leave evidence lists empty)
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{NOVELTY_5}

Be conservative in your ratings since you don't have related work context.
```

### `RATIONALE_NO_RELATED_BALANCED`

```
Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach
2. What aspects suggest the paper might be novel (leave evidence lists empty since no related papers are available)
3. What aspects suggest the paper might not be novel (leave evidence lists empty)
4. Your final assessment and conclusion about the paper's novelty
5. A novelty rating from 1 to 5:
{NOVELTY_5}

Rate the paper fairly based on its actual contributions. Use the full 1-5 scale:
- Rate 1-2 only if the paper clearly describes incremental or derivative work
- Rate 4-5 if the paper describes genuinely new ideas, methods, or insights
- Rate 3 for papers with moderate novelty or when uncertain
```

### `EVAL_SCALE`

```
Based on this, evaluate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

{CONSERVATIVE}

{RATIONALE_BASIC}
```

### `EVAL_SCALE_STRUCTURED`

```
Based on this, decide whether the paper is novel. It is novel if brings new ideas or develops new ideas previously unseen. Make sure that the ideas are truly unique. The paper is not novel if anything similar to it has been done before. Be very thorough. When in doubt, tend towards the not novel label.

{RATIONALE_STRUCTURED}
```

### `EVAL_SCALE_BALANCED`

```
Evaluate the paper's novelty based on its contributions compared to the related work.

IMPORTANT: The existence of related papers does NOT mean the paper lacks novelty. Related papers provide context, but novelty depends on whether this paper makes meaningful new contributions beyond what exists. A paper can be highly novel even if similar topics have been explored, as long as it offers significant new insights, methods, or perspectives.

Consider:
- Does the paper introduce new ideas, methods, or perspectives not present in related work?
- Does it solve problems in a substantially different or better way?
- Does it open new research directions or provide significant new insights?

Rate the paper fairly based on its actual contributions:
{NOVELTY_5}

{RATIONALE_STRUCTURED}
```

### `GRAPH_INTRO`

```
The paper summary describes the most important information about the paper and its contents. It summarises key aspects, which you can use to build a more comprehensive understanding of the paper.
```

### `RELATED_INTRO`

```
The related papers are split into "supporting" papers (those that corroborate the paper's ideas, methods, approach, etc.) and "contrasting" papers (those that go against the paper's ideas). Use these related papers to understand the context around the main paper, so you know what other works exist in comparison with the main paper.
```

### `RELATED_WITH_IDS`

```
Each related paper is presented with its ID, title, and relevant information. When referencing evidence from these papers, you MUST include the paper's ID and title.
```

---

## 1. Graph extraction

Source: `extract_graph.py`. Extracts a hierarchical entity graph (title,
primary_area, keywords, tldr, claim, method, experiment) from a paper.

### `full`

**System**

```
Given the following scientific paper, extract the important entities from the text and the relationships between them. The goal is to build a faithful and concise representation of the paper that captures the most important elements necessary to evaluate its novelty.
```

**Template**

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

### `nomethod`

**System**: same as `full`.

**Template** — drops the `method` entity; `claim` connects directly to `experiment`.

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
- `experiment`: what models, baselines, datasets, etc. were used in experiments to
support the `claims`.

Extract these entities and the relationships between them. The paper `title` is the root
entity. You must follow these rules when generating the entities and relationships:

- The `title` connects to the `primary_area`, the `keywords` and the `tldr` sentence.
- The `tldr` sentence connects to all the claims.
- Each `claim` must connect to one or more `experiments`.
- Every `experiment` must connected to at least one entity.
- Each `experiment` has a unique index starting from 0. These are indices from the
  `experiments` list in the graph. `claims` will connect to experiments by these
  indices.

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

### `noexperiments`

**System**: same as `full`.

**Template** — drops the `experiment` entity; `claim` connects to `method`.

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

Extract these entities and the relationships between them. The paper `title` is the root
entity. You must follow these rules when generating the entities and relationships:

- The `title` connects to the `primary_area`, the `keywords` and the `tldr` sentence.
- The `tldr` sentence connects to all the claims.
- Each `claim` must connect to one or more `methods`.
- Every `method` must connected to at least one entity.
- Each `method` has a unique index starting from 0. These are indices from the `methods`
  list in the graph. `claims` will connect to methods by these indices.

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

### `nodetail`

Same structure as `full` (kept as a no-detail variant for ablation labelling).
Template body identical to `full`.

### `excerpts`

**System**: same as `full`.

**Template** — same as `full` plus per-entity `detail` and verbatim `excerpts`.

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

## 2. Citation-context classification

Source: `classify_contexts.py`. Labels a citation context as `positive` or
`negative` polarity.

### `sentence`

```
You are given a citation context from a scientific paper that mentions. Your task is to determine the polarity of the citation context as 'positive' or 'negative'.

The polarity represents whether the citation context is supporting the paper's goals ('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Citation context: {context}
#####
Output:
```

### `simple`

```
You are given a main paper and a reference with a citation context. Your task is to determine the polarity of the citation context as 'positive' or 'negative', given the main paper's title, the reference's title, and the citation context where the main paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals ('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {main_title}
Reference title: {reference_title}
Citation context: {context}
#####
Output:
```

### `full`

```
You are given a main paper and a reference with a citation context. Your task is to
determine the polarity of the citation context as 'positive' or 'negative', given the
main paper's title and abstract, the reference's title and abstract, and the citation
context where the main paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals ('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {main_title}
Main paper abstract: {main_abstract}

Reference title: {reference_title}
Reference abstract: {reference_abstract}

Citation context: {context}
#####
Output:
```

---

## 3. Abstract splitting

Source: `abstract_classification.py`. Splits an abstract into background and
target sentences.

### `simple`

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

## 4. Related-paper summarisation (PETER-style)

Source: `summarise_related_peter.py`. Produces a short summary of how a related
paper supports/contrasts the main paper.

### `positive`

```
The following data contains information from a **Main Paper** and a **Related Paper**. The Related Paper has a positive relation to the Main Paper. It contains supporting information that strengthens the main paper claims.

Your task is to generate a summary that highlights how the Related Paper supports the Main Paper. Your summary should be short and concise, comprising a few sentences only.

#####
-Data-
# Main paper

**Title**: {title_main}
**Abstract**: {abstract_main}

# Related paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Output:
```

### `negative`

```
The following data contains information from a **Main Paper** and a **Related Paper**. The Related Paper has a negative relation to the Main Paper. It is used to contrast the claims made by the Main Paper.

Your task is to generate a summary that highlights how the Related Paper contrasts the Main Paper. Your summary should be short and concise, comprising of a few sentences only.

#####
-Data-
# Main paper

**Title**: {title_main}
**Abstract**: {abstract_main}

# Related paper

**Title**: {title_related}
**Abstract**: {abstract_related}

#####
Output:
```

### `positive-v2`

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

### `negative-v2`

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

## 5. Graph-based novelty evaluation

Source: `evaluate_graph.py`. Many variants combine: paper summary (graph),
related-paper sets, ablation toggles (sans related/graph), structured vs free
rationale, and balanced vs conservative scoring.

### `full-graph`

**System**

```
Given the following target paper, a summary and a selection of related papers separated by whether they're supporting or contrasting the main paper, give it a novelty rating.
```

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{EVAL_SCALE}

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
Output:
```

### `only-graph`

Same system and instruction blocks as `full-graph`. Drops `Title` and
`Abstract` from the data block — only the `approval` decision, graph summary,
and related papers are passed in.

### `title-graph`

Same as `full-graph` but the data block omits the abstract (only `Title`,
`Approval decision`, `Paper summary`, and related papers).

### `related`

**System**

```
Evaluate a paper's novelty based on its abstract and related work.
```

**Template**

```
You are evaluating a scientific paper's novelty based only on its abstract and related papers.

The related papers are split into:
- "Supporting" papers: work that builds on similar ideas or methods
- "Contrasting" papers: work that takes different approaches to similar problems

Read the abstract carefully to understand what the paper claims to contribute. Then check if any related papers describe the same contribution. Key points:

1. Focus on WHAT THE ABSTRACT CLAIMS as the contribution. Trust the authors' description of what they did.

2. Related papers provide context. Their existence shows this is an active research area, which is normal and expected. They only reduce novelty if they describe essentially the same contribution.

3. Topic similarity ≠ lack of novelty. Many papers study similar topics but make different contributions (e.g., different methods, applications, insights, or problem formulations).

4. Look for the specific claim: "We propose X", "We show Y", "We introduce Z". Does a related paper already propose/show/introduce that same X/Y/Z?

{RATIONALE_STRUCTURED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

### `related-structured`

```
The following data contains information about a scientific paper. It includes the main paper's title, abstract and some related papers. Note: No detailed paper summary is available for this evaluation.

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_BALANCED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

### `semantic-only`

**System**

```
Given the following target paper, a summary and semantically similar papers, provide a structured novelty evaluation on a 1-5 scale.
```

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and semantically similar papers found through search.

{GRAPH_INTRO}

The related papers were found through semantic similarity search. Note that semantic search may return papers that are only tangentially related or share surface-level similarities without being directly comparable work. Be cautious about using semantic matches as strong evidence against novelty - focus on whether the papers actually address the same core problem with the same approach.

{RELATED_WITH_IDS}

Evaluate the paper's novelty, but be aware that semantic similarity does not imply lack of novelty. A paper can be highly novel even if semantically similar papers exist, as long as its core contributions are meaningfully different.

{RATIONALE_STRUCTURED}

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

### `sans`

**System**

```
Given a paper's title and abstract, provide a structured novelty evaluation. This evaluation uses only the paper's own content without any related papers or graph summary.
```

**Template**

```
The following data contains information about a scientific paper. It includes only the paper's title and abstract.

Based solely on the title and abstract, evaluate the paper's novelty. Consider:
- What problem or question is being addressed?
- What methods or approaches are proposed?
- How significant do the contributions appear to be?

You do not have access to related papers, so base your assessment purely on how the paper presents its contributions and how novel the described approach sounds.

{RATIONALE_NO_RELATED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

#####
```

### `sans-balanced`

Same system and template as `sans` but using `RATIONALE_NO_RELATED_BALANCED`
(removes conservative bias).

### `norel-graph`

**System**

```
Given the following target paper and its graph summary, provide a structured novelty evaluation based solely on the paper's own content.
```

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, abstract, and a summary of its key points extracted as a graph.

{GRAPH_INTRO}

Note: This evaluation does NOT include related papers - only the main paper's content and graph summary are available.

Based solely on the paper's content and graph summary, evaluate how novel the described approach and contributions appear to be.

{RATIONALE_NO_RELATED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Paper summary:
{graph}

#####
```

### `full-graph-positive`

**System**

```
Given the following target paper, a summary and a selection of related papers supporting the main paper, give it a novelty rating.
```

**Template** — supporting papers only.

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

The related papers support the main paper, i.e. they corroborate the paper's ideas, methods, approach, etc. Use these related papers to understand the context around the main paper, so you know what other works exist in comparison with the main paper.

{EVAL_SCALE}

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

#####
Output:
```

### `full-graph-negative`

**System**

```
Given the following target paper, a summary and a selection of related papers contrasting the main paper, give it a novelty rating.
```

**Template** — contrasting papers only (uses the same data block layout as
`full-graph`).

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

The related papers contrast the main paper, i.e. they go against the main paper's ideas. Use these related papers to understand the context around the main paper, so you know what other works exist in comparison with the main paper.

{EVAL_SCALE}

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
Output:
```

### `attribution-graph`

**System**: same as `full-graph`.

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

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

Based on this, evaluate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

Be very thorough. When in doubt, tend towards lower ratings.

For each related paper, assign a code to them. When discussing related papers in the rationale, be specific and use the paper's code. After the rationale, add a references section with the code and name for each paper. For brevity, don't use the paper names in the rationale, only the code. Make sure that all referenced papers appear in the references section.

First, generate the rationale for your novelty rating, then give the final rating (1-5).
```

### `full-graph-basic`

**System**

```
Given the following target paper, a summary and a selection of related papers separated by whether they're supporting or contrasting the main paper, give it a novelty rating on a 1-5 scale.
```

**Template**: identical body to `full-graph` (free-form rating via `EVAL_SCALE`).

### `full-graph-structured`

**System**

```
Given the following target paper, a summary and a selection of related papers separated by whether they're supporting or contrasting the main paper, provide a structured novelty evaluation on a 1-5 scale.
```

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_STRUCTURED}

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

### `full-graph-balanced`

**System**

```
Given the following target paper, a summary and a selection of related papers separated by whether they're supporting or contrasting the main paper, provide a structured novelty evaluation.
```

**Template**

```
The following data contains information about a scientific paper. It includes the main paper's title, a summary of its key points and some related papers.

{GRAPH_INTRO}

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_BALANCED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

Paper summary:
{graph}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

### `debug-random`

Debug variant — assigns a random label.

```
The following data contains information about a scientific paper. It includes the paper's title and abstract.

Assign a random label between 1 and 5. Your rationale should be "debug".

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
```

### `simple-basic`

```
The following data contains information about a paper that needs novelty assessment. It includes the paper's title and abstract, along with related papers discovered through semantic search.

The related papers are ordered by relevance and include both supporting papers (those with similar research directions or methods) and contrasting papers (those with different approaches to similar problems). Use these related papers to understand the research landscape and assess whether the target paper presents novel contributions.

{RELATED_WITH_IDS}

Based on this information, evaluate the paper's novelty on a 1-5 scale. Be thorough in your assessment and when in doubt, tend towards lower ratings.

Provide your evaluation in structured format with:
1. A brief summary of the paper's main contributions and approach based on the title and abstract
2. Specific evidence from related papers that supports the paper's novelty (up to 5 items)
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that supports novelty
   - Indicate that this is a search-based related paper
3. Specific evidence from related papers that contradicts the paper's novelty (up to 5 items)
   - For each piece of evidence, include the paper ID and title it comes from
   - Quote or paraphrase the specific finding that contradicts novelty
   - Indicate that this is a search-based related paper
4. Key technical comparisons that influenced your decision
5. Your final assessment and conclusion about the paper's novelty
6. A novelty rating from 1 to 5:
{NOVELTY_5}

IMPORTANT: When creating evidence items, structure them as:
- text: The evidence text describing the finding
- paper_id: The ID of the paper (if available)
- paper_title: The title of the paper
- source: "search" to indicate this paper was found through semantic search

#####
{demonstrations}

-Data-
Title: {title}

Abstract: {abstract}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

### `full-text`

**System**

```
Given the full text of a scientific paper and a selection of related papers, provide a structured novelty evaluation on a 1-5 scale.
```

**Template**

```
The following data contains the full text of a scientific paper and some related papers.

{RELATED_INTRO}

{RELATED_WITH_IDS}

{EVAL_SCALE_BALANCED}

#####
{demonstrations}

-Data-
Title: {title}

Full paper text:
{main_text}

Supporting papers:
{positive}

Contrasting papers:
{negative}

#####
```

### `full-text-only`

**System**

```
Given the full text of a scientific paper, provide a structured novelty evaluation on a 1-5 scale. No related papers are available.
```

**Template**

```
The following data contains the full text of a scientific paper.

Based solely on the paper's content, evaluate how novel the described approach and
contributions appear to be. You do not have access to related papers, so base your
assessment purely on how the paper presents its contributions.

{RATIONALE_NO_RELATED_BALANCED}

#####
{demonstrations}

-Data-
Title: {title}

Full paper text:
{main_text}

#####
```

### `sc4anm`

**System**

```
You are a scientific paper reviewer. Based on the following sections of an academic paper, predict its novelty score.
```

**Template**

```
The following data contains key sections (Introduction, Results, Discussion) of a scientific paper. These are the sections most relevant to assessing the paper's novelty.

Based solely on the paper's section content, evaluate its novelty. You do not have access to related papers, so base your assessment purely on how the paper presents its contributions.

{RATIONALE_NO_RELATED_BALANCED}

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}

{ird_sections}

#####
```

### `rag-novelty`

**System**

```
You are a scientific paper reviewer assessing the novelty of a research paper. You have been provided with several related papers retrieved from the literature to help inform your assessment.
```

**Template**

```
The following data contains a target paper and related papers retrieved by similarity to the target paper's abstract. Use the related papers to inform your novelty assessment.

{EVAL_SCALE_BALANCED}

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

## 6. SciMON novelty evaluation

Source: `evaluate_paper_scimon.py`. Rates novelty using inspiration sentences
(graph-extracted terms) from related papers.

### `simple`

**System**

```
Given inspiration sentences from related papers, give a novelty rating to a paper submitted to a high-quality scientific conference on a 1-5 scale.
```

**Template**

```
The following data contains information about a scientific paper. It includes the paper's title and abstract, and some inspiration sentences from related papers. These sentences are meant to aid you in understanding whether the ideas in the paper are novel.

{EVAL_SCALE}

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

## 7. NovAScore ACU extraction

Source: `extract_acu.py`. Extracts atomic content units (ACUs) — single,
indivisible factual claims — from a paper.

### `simple`

Original NovAScore-paper prompt.

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

### `sci`

Scientific-paper-tuned variant with multiple demonstrations.

```
Extract atomic content units (ACUs) from this scientific paper. An ACU represents a single, indivisible fact or claim from the paper.

For scientific papers, good ACUs should:
1. Capture specific scientific findings, methodologies, or conclusions
2. Preserve precise technical details, measurements, and relationships
3. Maintain scientific accuracy without simplification
4. Use the same technical terminology as the original text
5. Include all relevant entities, values, and conditions for each claim
6. Be standalone statements that can be individually verified

When extracting ACUs:
- Replace pronouns with their specific referents
- Preserve numeric values and units exactly as stated
- Maintain relationships between variables, conditions, and outcomes
- Include methodology details when relevant to findings
- Separate compound statements into individual claims

#####
-Demonstrations-

--Example document 1--

Enzon Inc. said tests of a new anti-cancer drug to treat a form of leukemia showed positive results, Abraham Abuchowski, president of the pharmeutical company, told the annual meeting. Enzon also said it filed its first new drug application with the FDA for an orphan drug that treats severe combined immunodeficiency disease, or SCID. The company has about 10.8 million common shares outstanding. Eastman Kodak Co. ownas an 18% interest in Enzon.

Example ACUs:
- Enzon Inc. conducted tests of a new anti-cancer drug for treating a form of leukemia.
- Tests of Enzon Inc.'s new anti-cancer drug for leukemia showed positive results.
- Abraham Abuchowski is the president of Enzon Inc.
- Abraham Abuchowski reported the positive test results at the annual meeting.
- Enzon Inc. filed its first new drug application with the FDA.
- The new drug application filed by Enzon Inc. with the FDA is for an orphan drug.
- The orphan drug developed by Enzon Inc. treats severe combined immunodeficiency disease (SCID).
- Enzon Inc. has approximately 10.8 million common shares outstanding.
- Eastman Kodak Co. owns an 18% interest in Enzon Inc.

--Example document 2--

In a randomized controlled trial, researchers investigated the effects of a novel SGLT2 inhibitor on HbA1c levels in patients with type 2 diabetes. 128 participants were administered either 100mg of the inhibitor or a placebo daily for 12 weeks. Results showed that the treatment group (n=64) experienced a mean reduction in HbA1c of 1.8% (95% CI: 1.5-2.1%), compared to 0.4% (95% CI: 0.2-0.6%) in the control group (p<0.001). Additionally, the treatment group showed significant improvements in fasting plasma glucose (-2.3 mmol/L vs -0.6 mmol/L, p<0.001) and body weight (-2.7 kg vs -0.5 kg, p<0.01). Seven patients (10.9%) in the treatment group reported mild gastrointestinal side effects, compared to three patients (4.7%) in the placebo group.

Example ACUs:
- Researchers conducted a randomized controlled trial to investigate the effects of a novel SGLT2 inhibitor on HbA1c levels.
- The study focused on patients with type 2 diabetes.
- The trial included 128 participants in total.
- 64 participants received 100mg of the SGLT2 inhibitor daily.
- 64 participants received a placebo daily.
- The treatment duration was 12 weeks.
- The treatment group experienced a mean reduction in HbA1c of 1.8%.
- The 95% confidence interval for HbA1c reduction in the treatment group was 1.5-2.1%.
- The control group experienced a mean reduction in HbA1c of 0.4%.
- The 95% confidence interval for HbA1c reduction in the control group was 0.2-0.6%.
- The difference in HbA1c reduction between groups was statistically significant (p<0.001).
- The treatment group showed a reduction in fasting plasma glucose of 2.3 mmol/L.
- The control group showed a reduction in fasting plasma glucose of 0.6 mmol/L.
- The difference in fasting plasma glucose reduction was statistically significant (p<0.001).
- The treatment group showed a reduction in body weight of 2.7 kg.
- The control group showed a reduction in body weight of 0.5 kg.
- The difference in body weight reduction was statistically significant (p<0.01).
- Seven patients (10.9%) in the treatment group reported mild gastrointestinal side effects.
- Three patients (4.7%) in the placebo group reported mild gastrointestinal side effects.

-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
```

---

## 8. Search evaluation

Source: `evaluate_search.py`. Uses LLM web search to retrieve related work,
then assigns a novelty rating.

### `simple`

**System**

```
Given the following target paper, search the internet for relevant publications, and based on them, give a novelty rating from 1 to 5.
```

**Template**

````
The following data contains information about a scientific paper. It includes the paper's title and abstract.

First, search the web for publications related to the paper. Your goal is to find relevant papers to compare the main paper with. This is important to determine if the paper's contributions are novel.

Based on this content, evaluate the paper's novelty. This should reflect how much the paper brings and develops new ideas previously unseen in the literature.

Rate the paper's novelty on a 1-5 scale:
{NOVELTY_5}

Be thorough. When in doubt, tend towards lower ratings.

First generate the rationale for your novelty rating, then give the final rating (1-5).

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

### `attribution`

**System**: same as `simple`.

**Template** — same as `simple` plus this attribution instruction before the
output format block:

```
The rationale must include the documents retrieved by web search. It must be pure plain text without any formatting. Instead of writing the titles and links to the documents inside the rationale, assign each a number and list them (number, title and link) at the bottom of the text.
```

---

## 9. Pairwise rationale comparison

Source: `pairwise_comparison.py`. Picks the better of two rationales (A vs B)
for a given evaluation metric.

### `standard`

**System**

```
You are evaluating and comparing two different rationales (labeled A and B) for the same scientific paper.
Your job is to determine which rationale is better according to a specific evaluation metric.
```

**Template**

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
