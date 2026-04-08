# SC4ANM Baseline Implementation

We reimplement the GPT prompting path from SC4ANM (Wu et al., 2025), which segments papers into IMRaD sections and prompts an LLM with the optimal section combination (Introduction + Results + Discussion) for novelty prediction. We skip the PLM fine-tuning path (Longformer/BigBird) since it requires training on their dataset and the GPT variant is more directly comparable to our other LLM-based baselines.

**Citation**: Wu, W., Zhang, C., Bao, T., & Zhao, Y. (2025). SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Prediction in Academic Papers. *Expert Systems with Applications*. Code: `github.com/njust-winchy/SC4ANM`.

## Approach (for paper write-up)

### What we implemented

We reimplemented the GPT prompting path from SC4ANM. The original method has two key stages: (1) classify paper text into IMRaD (Introduction, Methods, Results, Discussion) sections, and (2) prompt an LLM with the section combination found optimal in their experiments — Introduction + Results + Discussion (IRD).

### Section classification

SC4ANM uses a pre-trained SciBERT classifier (`howanching-clara/classifier_for_academic_texts` on HuggingFace) to classify sentences/paragraphs into IMRaD categories. Since our SciNova and PeerRead papers are already parsed from LaTeX/PDF into structured Markdown with section headings preserved, we replace the classifier with **heading-based heuristic matching**, which is more reliable for well-structured papers:

| IMRaD Category | Heading patterns (case-insensitive regex)                                           |
|----------------|-------------------------------------------------------------------------------------|
| Introduction   | `introduction`                                                                      |
| Methods        | `method`, `methodology`, `approach`, `framework`, `model`, `proposed`, `technique`  |
| Results        | `result`, `experiment`, `evaluation`, `empirical`, `analysis`                       |
| Discussion     | `discussion`, `conclusion`, `limitation`, `future.work`, `summary`                  |
| **Excluded**   | `related.work`, `background`, `preliminary`, `appendix`, `acknowledge`, `reference` |

Only the **IRD combination** is used (no Methods), following SC4ANM's finding that this combination is optimal for novelty prediction.

**Heading coverage across our datasets** (from audit):

| Category     | ORC 100 | PeerRead 68 | Novelty-only 87 |
|--------------|---------|-------------|-----------------|
| Introduction | 99%     | 97%         | 99%             |
| Results      | 88%     | 93%         | 89%             |
| Discussion   | 95%     | 99%         | 97%             |
| Full I+R+D   | 86%     | 91%         | 87%             |

Papers missing a section receive "[Section not available]" in its place.

### Truncation

Following SC4ANM §4, each section is truncated to a maximum of 2,000 tokens using the `cl100k_base` tiktoken encoding, keeping total input under ~8K tokens including the prompt template.

### Prompt

The prompt follows SC4ANM's GPT evaluation setup. The LLM receives the paper title plus the three IRD sections, described as "the sections most relevant to assessing the paper's novelty", and is asked to rate novelty on the same 1–5 scale used across all our experiments. The prompt does not reference SC4ANM or any methodology by name. We use GPTStructured output (matching our other balanced prompts) rather than SC4ANM's free-text "Rationale: ... Score: ..." format, for consistency with our evaluation pipeline.

### Model and runs

- **Model**: GPT-4o-mini (matching our other LLM baselines).
- **Runs**: 3 runs per experiment for variance estimation (same as all other baselines).
- **Temperature**: 0 (deterministic).

### Key differences from our other baselines

| Aspect            | SC4ANM         | Basic (sans)  | Full-text-only  | GraphMind              |
|-------------------|----------------|---------------|-----------------|------------------------|
| Input text        | I+R+D sections | Abstract only | Full paper text | Graph + related papers |
| Related papers    | No             | No            | No              | Yes (PETER)            |
| Section selection | Optimal (IRD)  | N/A           | All sections    | N/A                    |
| Token budget      | ~6K (3 × 2K)   | ~500          | Unbounded       | Varies                 |

SC4ANM sits between the abstract-only baseline (less text) and the full-text baseline (all text), using a principled section selection strategy.

## Implementation details

### Source files

| File                                      | Purpose                                                               |
|-------------------------------------------|-----------------------------------------------------------------------|
| `src/paper/baselines/sc4anm.py`           | IMRaD heading classification, token truncation, IRD formatting        |
| `src/paper/gpt/prompts/evaluate_graph.py` | SC4ANM prompt template (`sc4anm` key)                                 |
| `src/paper/gpt/evaluate_paper_graph.py`   | Wiring: populates `{ird_sections}` variable in `format_eval_template` |

### How it works

1. **`classify_sections()`** iterates paper sections, matches each heading against regex patterns, and buckets text into IMRaD categories. Excluded headings (related work, background, appendix, etc.) are skipped. Multiple sections matching the same category are concatenated.

2. **`format_ird_sections()`** takes the classified sections, truncates each to 2,000 tokens via `gpt.tokenizer.truncate_text` (cl100k_base), and formats them as labelled blocks. Missing sections become "[Section not available]".

3. **`format_sc4anm_template()`** (in `evaluate_paper_graph.py`) calls both functions and passes the result as the `{ird_sections}` template variable. This is a dedicated format function, separate from `format_eval_template`, dispatched when `--eval-prompt sc4anm` is selected.

4. The prompt uses `GPTStructured` output type (same as `full-text-only` and `sans-balanced`), which produces a structured evaluation with rationale + rating.

### Running experiments

```bash
# Single run (ORC 100)
uv run paper gpt eval graph run \
  --papers output/venus5/split/dev_100_balanced.json.zst \
  --output output/baselines/sc4anm/orc_100 \
  --model gpt-4o-mini \
  --eval-prompt sc4anm \
  --limit 100

# Experiment with 3 runs (ORC 100)
uv run paper gpt eval graph experiment \
  --papers output/venus5/split/dev_100_balanced.json.zst \
  --output output/baselines/sc4anm/orc_100_experiment \
  --model gpt-4o-mini \
  --eval-prompt sc4anm \
  --limit 100 \
  --runs 3

# PeerRead 68
uv run paper gpt eval graph experiment \
  --papers output/new_peerread/peter_summarised/balanced_68.json.zst \
  --output output/baselines/sc4anm/peerread_68_experiment \
  --model gpt-4o-mini \
  --eval-prompt sc4anm \
  --limit 70 \
  --runs 3

# Novelty-only 87
uv run paper gpt eval graph experiment \
  --papers output/venus5/split/novelty_only_87_balanced.json.zst \
  --output output/baselines/sc4anm/novelty_87_experiment \
  --model gpt-4o-mini \
  --eval-prompt sc4anm \
  --limit 87 \
  --runs 3
```

No graph extraction occurs (the pipeline skips it since "graph" is not in the prompt name). No training or external data is required.
