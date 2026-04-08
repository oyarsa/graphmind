# RAG-Novelty Baseline Implementation

We reimplement RAG-Novelty from Lin, Peng, and Fang (AISD@NAACL 2025). The original method embeds paper abstracts, retrieves top-K similar papers from a corpus, and uses them as context for novelty scoring. We adapt it from pairwise comparison to pointwise 1–5 scoring.

**Citation**: Lin, E., Peng, Z., & Fang, Y. (2025). Evaluating and Enhancing Large Language Models for Novelty Assessment in Scholarly Publications. *Proceedings of the 1st Workshop on AI and Scientific Discovery (AISD@NAACL 2025)*, pp. 46–57. Code: `github.com/ethannlin/SchNovel`.

## Approach (for paper write-up)

### What we implemented

We reimplemented the RAG-Novelty prompting pipeline. The original method has three stages: (1) embed paper abstracts using an embedding model, (2) retrieve top-K similar papers from a pre-built corpus via cosine similarity, and (3) prompt an LLM with the target paper and retrieved papers for novelty scoring.

### Retrieval: reusing PETER-collected related papers

Rather than building an independent retrieval corpus with new embeddings, we reuse the related papers already collected via PETER (citations + Semantic Scholar semantic neighbours). This gives us a controlled comparison with GraphMind: **both methods see the same pool of related papers**, but present them differently to the LLM. RAG-Novelty uses a flat ranked list; GraphMind structures them by citation polarity and pairs them with a hierarchical paper graph.

We use **all** of each paper's PETER-collected related papers (typically 15–20), sorted by similarity score and flattened across citation and semantic sources (ignoring polarity). Since PETER's papers were already selected for relevance, they correspond to what RAG-Novelty's embedding retrieval would have surfaced from a larger corpus. They are presented as a single ranked list without source or polarity distinctions.

### Average publication year heuristic

A distinctive feature of RAG-Novelty is the "average published dates" heuristic. The LLM is told to consider the average publication year of the retrieved papers as additional context: papers retrieving more recent related work may themselves be more novel within their field. We retain this heuristic.

Note: publication year is available for PeerRead papers but not for ORC papers (where `year` is null). When no year data is available, the heuristic is silently omitted from the prompt.

### Self-reflection instruction

The prompt includes a self-reflection step from RAG-Novelty's best-performing prompt variant: after forming an initial assessment, the LLM is asked to reflect on whether it may be over- or under-estimating novelty. This is retained from the original paper.

### Prompt

The LLM receives the paper title, abstract, and all PETER-collected related papers (each with title, year, and abstract), sorted by similarity score. The prompt uses `EVAL_SCALE_BALANCED` (the same balanced evaluation scale used for our GraphMind structured prompts), which includes structured evidence fields that reference the related papers. The self-reflection instruction and average-year heuristic are appended.

We use `GPTStructured` output type for consistency with our evaluation pipeline.

### Model and runs

- **Model**: GPT-4o-mini (matching our other LLM baselines).
- **Runs**: 3 runs per experiment for variance estimation (same as all other baselines).
- **Temperature**: 0 (deterministic).

### Key differences from our other baselines

| Aspect              | RAG-Novelty              | Basic (sans)  | SC4ANM         | GraphMind                       |
|---------------------|--------------------------|---------------|----------------|---------------------------------|
| Input text          | Abstract                 | Abstract only | I+R+D sections | Graph + related papers          |
| Related papers      | All, flat ranked list    | No            | No             | Polarity-split + summaries      |
| Paper presentation  | Title + abstract + year  | N/A           | N/A            | Structured graph + summaries    |
| Retrieval method    | Score-ranked (flattened)  | N/A           | N/A            | Citation + semantic (PETER)     |
| Special heuristics  | Average year, self-reflection | None    | None           | None                            |

RAG-Novelty uses the same related papers as GraphMind but presents them as a flat list with no polarity or structural information. This isolates the value of GraphMind's structured processing.

## Implementation details

### Source files

| File                                      | Purpose                                                                    |
|-------------------------------------------|----------------------------------------------------------------------------|
| `src/paper/baselines/rag_novelty.py`      | Top-K selection, average year computation, retrieved papers formatting      |
| `src/paper/gpt/prompts/evaluate_graph.py` | RAG-Novelty prompt template (`rag-novelty` key)                            |
| `src/paper/gpt/evaluate_paper_graph.py`   | `format_rag_novelty_template` + dispatch in `build_eval_prompt_text`       |

### How it works

1. **`format_retrieved_papers()`** formats the papers as a numbered list: "Paper 1: {title} ({year})\nAbstract: {abstract}".

2. **`format_rag_novelty_context()`** sorts all related papers by score (descending), formats them, and computes the average publication year when available. Returns (papers_text, year_context) ready for template substitution.

3. **`format_rag_novelty_template()`** (in `evaluate_paper_graph.py`) calls `format_rag_novelty_context` and populates the prompt's `{retrieved_papers}`, `{year_context}`, `{title}`, `{abstract}`, and `{demonstrations}` variables.

4. **`build_eval_prompt_text()`** dispatches to `format_rag_novelty_template` when `--eval-prompt rag-novelty` is selected.

### Running experiments

```bash
# ORC 100
uv run paper gpt eval graph experiment \
  --papers output/venus5/split/dev_100_balanced.json.zst \
  --output output/baselines/rag_novelty/orc_100_experiment \
  --model gpt-4o-mini \
  --eval-prompt rag-novelty \
  --limit 100 \
  --runs 3

# PeerRead 68
uv run paper gpt eval graph experiment \
  --papers output/new_peerread/peter_summarised/balanced_68.json.zst \
  --output output/baselines/rag_novelty/peerread_68_experiment \
  --model gpt-4o-mini \
  --eval-prompt rag-novelty \
  --limit 70 \
  --runs 3

# Novelty-only 87
uv run paper gpt eval graph experiment \
  --papers output/venus5/split/novelty_only_87_balanced.json.zst \
  --output output/baselines/rag_novelty/novelty_87_experiment \
  --model gpt-4o-mini \
  --eval-prompt rag-novelty \
  --limit 87 \
  --runs 3
```

No embedding, retrieval, or training is required. The pipeline reuses existing PETER-collected related papers.
