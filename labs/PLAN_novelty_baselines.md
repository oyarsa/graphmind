# Experiment Plan: Novelty-Only Subset + New Baselines

## Progress

### Build
- [x] 1.1 Create novelty-only balanced subset (87 papers)
- [x] 1.2 Raw-abstracts baseline (swap summary for abstract)
- [x] 1.3 Full paper content prompt variant
- [x] 1.4 Download CSAbstruct dev set
- [x] 2.0 Create SFT input modes and configs for full-text and full-text+abstracts

### Finetuning (GPU required)
- [ ] 2.1 Finetune Llama with full paper text
- [ ] 2.2 Finetune Llama with full paper text + retrieved abstracts
- [ ] 2.3 Finetune Qwen with full paper text
- [ ] 2.4 Finetune Qwen with full paper text + retrieved abstracts

### Novelty-only experiments
- [x] 3.1 NovaSCORE — Pearson 0.158
- [x] 3.2 SciMON — Pearson 0.287 ± 0.013 (82/87 papers)
- [x] 3.11 GraphMind GPT-4o-mini — Pearson 0.508 ± 0.036
- [x] 3.12 GraphMind Gemini — Pearson 0.497 ± 0.027
- [x] 3.14 Abs. + Related G. — Pearson 0.021 ± 0.054 (rerun with v6 prompt)
- [x] 3.18 GraphMind (same as 3.11)
- [x] 3.19 Full paper content — Pearson 0.061 ± 0.096

### New baselines on main datasets
- [x] 4.1 Raw-abstracts (ORC 100) — Pearson 0.024 ± 0.077
- [x] 4.2 Raw-abstracts (PeerRead 68) — Pearson 0.001 ± 0.095
- [x] 4.3 Full paper content (ORC 100) — Pearson -0.010 ± 0.117
- [x] 4.4 Full paper content (PeerRead 68) — Pearson -0.001 ± 0.119

All four baselines produce near-zero or negative correlation. Raw abstracts are worse than
GPT summaries. Full paper text is the most expensive with no benefit.

### Finetuned model evaluation (novelty-only)
- [x] 5.0 Prepare novelty-only data for SFT inference (basic + graph-enriched)
- [ ] 5.1-5.4 Existing checkpoints (Llama/Qwen basic+graph) — BLOCKED: fleche in use by Phase 2
- [ ] 5.5-5.8 New checkpoints (Llama/Qwen full-text+related) — BLOCKED: Phase 2 training

### CSAbstruct
- [x] 6.0 Download CSAbstruct dev set (295 entries)
- [x] 6.1 Build evaluation script
- [x] 6.2 Run LLM classification on 291 abstracts (4 used as demos)
- [x] 6.3 Compute text-overlap metrics — ROUGE-L: 0.702 bg / 0.840 tgt, BERTScore: 0.869 / 0.963
- [x] 6.4 Compute sentence-level metrics — Accuracy: 84.5%, F1: 0.785 bg / 0.878 tgt
- [x] 6.5 Log results

### TBD (not yet requested)
- [ ] 3.3-3.6 Basic Prompting (Llama, Qwen, GPT-4o-mini, Gemini)
- [ ] 3.7-3.8 Search-Augmented (GPT-4o-mini, Gemini)
- [ ] 3.9-3.10 GraphMind (Llama, Qwen)
- [ ] 3.13 Abstract only ablation
- [ ] 3.15 H.G. only ablation
- [ ] 3.16 H.G. + Citation ablation
- [ ] 3.17 H.G. + Semantic ablation

## Decisions

- **Novelty-only subset**: ICLR 2022-2023 from ORC (uses
  `technical_novelty_and_significance` / `empirical_novelty_and_significance` instead of
  the broader `contribution` field).
- **Splits used**: dev+test combined (deviates from main experiments' dev-only convention
  because ICLR 2022-2023 has very few rating-4 papers in dev alone -- only 7).
- **Rating 1 dropped**: Only 2 papers with rating 1 in dev+test. Balanced on ratings 2-4.
- **Balanced subset size**: 29 per class x 3 classes = **87 papers**.
- **Stratification**: Papers can be stratified by `paper.conference` (e.g. `iclr2022`,
  `iclr2023`) and by primary area (extracted graph entity, 20 categories).

## Phase 1: Build prerequisites

### 1.1 Create novelty-only balanced subset

Filter ORC dev+test to ICLR 2022-2023, drop rating 1, balance on ratings 2-4 (29/class).
Output: `output/venus5/split/novelty_only_87_balanced.json.zst`

### 1.2 Raw-abstracts baseline

Modify `format_related()` in `src/paper/gpt/evaluate_paper_graph.py` to use
`paper.abstract` instead of `paper.summary`. Add a new `--eval-prompt` variant or a flag
to toggle this behaviour.

### 1.3 Full paper content prompt variant

Create a new eval prompt that passes `main_text()` as the paper representation, replacing
graph and abstract entirely. Retrieved papers remain as-is (summaries, or raw abstracts
once 1.2 is done).

### 1.4 Download CSAbstruct evaluation set

```bash
uv run paper demonstrations abstract output/csabstruct_dev.json --entries 295 --split dev
```

## Phase 2: Finetuning (GPU required)

Added `InputMode` enum to `sft_gen.py` with three modes: `basic`, `full-text`,
`full-text-abstracts`. Controlled by `input_mode` in TOML config. All 4 configs created.

Token length measurements (from ORC dev 100 balanced):
- Main text: mean=20,642 median=16,228 p90=45,936 max=78,306
- Related abstracts: mean=3,948 median=3,904
- Combined: mean=24,589 median=20,278
- Config `max_length=8192` -- significant truncation expected, but GPU memory constrains us.

### Training commands

```bash
# 2.1 Llama full-text
uv run paper baselines gen train \
  --train output/baselines/llama_data/orc_train.json.zst \
  --dev output/baselines/llama_data/orc_dev.json.zst \
  --test output/baselines/llama_data/orc_test.json.zst \
  --output output/baselines/llama_gen_fulltext \
  --config src/paper/baselines/sft_config/llama_gen_fulltext.toml

# 2.2 Llama full-text + abstracts
uv run paper baselines gen train \
  --train output/baselines/llama_data/orc_train.json.zst \
  --dev output/baselines/llama_data/orc_dev.json.zst \
  --test output/baselines/llama_data/orc_test.json.zst \
  --output output/baselines/llama_gen_fulltext_abstracts \
  --config src/paper/baselines/sft_config/llama_gen_fulltext_abstracts.toml

# 2.3 Qwen full-text
uv run paper baselines gen train \
  --train output/baselines/llama_data/orc_train.json.zst \
  --dev output/baselines/llama_data/orc_dev.json.zst \
  --test output/baselines/llama_data/orc_test.json.zst \
  --output output/baselines/qwen_gen_fulltext \
  --config src/paper/baselines/sft_config/qwen_gen_fulltext.toml

# 2.4 Qwen full-text + abstracts
uv run paper baselines gen train \
  --train output/baselines/llama_data/orc_train.json.zst \
  --dev output/baselines/llama_data/orc_dev.json.zst \
  --test output/baselines/llama_data/orc_test.json.zst \
  --output output/baselines/qwen_gen_fulltext_abstracts \
  --config src/paper/baselines/sft_config/qwen_gen_fulltext_abstracts.toml
```

## Phase 3: Experiments on novelty-only subset (87 papers)

All experiments below run on `novelty_only_87_balanced.json.zst`.

### Method comparison

| #    | Category         | Model       | Status  | Notes                                   |
|------|------------------|-------------|---------|-----------------------------------------|
| 3.1  | Existing Methods | NovaSCORE   | Planned | Need ACU data for novelty-only papers   |
| 3.2  | Existing Methods | SciMON      | Planned | Need SciMON graph for novelty-only      |
| 3.3  | Basic Prompting  | Llama       | TBD     | Existing checkpoint, `sans` prompt      |
| 3.4  | Basic Prompting  | Qwen        | TBD     | Existing checkpoint, `sans` prompt      |
| 3.5  | Basic Prompting  | GPT-4o-mini      | TBD     | `--eval-prompt sans --model gpt-4o-mini`     |
| 3.6  | Basic Prompting  | Gemini      | TBD     | `--eval-prompt sans --model gemini-2.0-flash` |
| 3.7  | Search-Augmented | GPT-4o-mini      | TBD     | `--eval-prompt related-structured --model gpt-4o-mini` |
| 3.8  | Search-Augmented | Gemini      | TBD     | `--eval-prompt related-structured --model gemini-2.0-flash` |
| 3.9  | GraphMind        | Llama       | TBD     | Existing checkpoint, graph input        |
| 3.10 | GraphMind        | Qwen        | TBD     | Existing checkpoint, graph input        |
| 3.11 | GraphMind        | GPT-4o-mini      | Planned | `--eval-prompt full-graph-structured --model gpt-4o-mini`, 5 runs |
| 3.12 | GraphMind        | Gemini      | Planned | `--eval-prompt full-graph-structured --model gemini-2.0-flash`, 5 runs |

### Ablation / configuration variants

| #    | Variant          | Status  | Eval prompt              | Sources   |
|------|------------------|---------|--------------------------|-----------|
| 3.13 | Abstract only    | TBD     | `sans`                   | N/A       |
| 3.14 | Abs. + Related G.| Planned | `related-structured`     | both      |
| 3.15 | H.G. only        | TBD     | `norel-graph`            | N/A       |
| 3.16 | H.G. + Citation  | TBD     | `full-graph-structured`  | citations |
| 3.17 | H.G. + Semantic  | TBD     | `semantic-only`          | semantic  |
| 3.18 | GraphMind        | Planned | `full-graph-structured`  | both      |

### New baselines

| #    | Variant             | Status  | Notes                    |
|------|---------------------|---------|--------------------------|
| 3.19 | Full paper content  | Planned | New prompt from 1.3      |

## Phase 4: New baselines on main datasets

Run on existing balanced sets (ORC 100, PeerRead 68) for comparison with prior results.

| #   | Experiment                   | Dataset     |
|-----|------------------------------|-------------|
| 4.1 | Raw-abstracts baseline (1.2) | ORC 100     |
| 4.2 | Raw-abstracts baseline (1.2) | PeerRead 68 |
| 4.3 | Full paper content (1.3)     | ORC 100     |
| 4.4 | Full paper content (1.3)     | PeerRead 68 |

## Phase 5: Evaluate finetuned models on novelty-only subset

Evaluate all existing + new checkpoints (from Phase 2) on the novelty-only 87-paper set.

| #   | Model                                   | Input mode                            |
|-----|-----------------------------------------|---------------------------------------|
| 5.1 | Llama basic (existing)                  | title + abstract                      |
| 5.2 | Llama graph (existing)                  | title + abstract + graph + related    |
| 5.3 | Qwen basic (existing)                   | title + abstract                      |
| 5.4 | Qwen graph (existing)                   | title + abstract + graph + related    |
| 5.5 | Llama full-text (new, from 2.1)         | full paper text                       |
| 5.6 | Llama full-text+related (new, from 2.2) | full paper text + retrieved abstracts |
| 5.7 | Qwen full-text (new, from 2.1)          | full paper text                       |
| 5.8 | Qwen full-text+related (new, from 2.2)  | full paper text + retrieved abstracts |

### Data (ready)

- `output/baselines/llama_data/novelty_test.json.zst` — basic models input
- `output/baselines/llama_data/novelty_test_graph_enriched.json.zst` — graph models
  input (87 graphs from cache, zero API cost)

### Blockers

- **5.1-5.4 (existing checkpoints)**: Fleche is in use by Phase 2 training. Can run
  as soon as the training jobs finish.
- **5.5-5.8 (new checkpoints)**: Need Phase 2 training to complete first.

### Inference commands (existing checkpoints, 5 seeds each)

```bash
# 5.1 Llama basic (seeds 42-46)
for seed in 42 43 44 45 46; do
  fleche run infer_gen --env DATASET=novelty \
    --env CONFIG=llama_orc_gen_seed${seed}
done

# 5.2 Llama graph (seeds 42-46)
for seed in 42 43 44 45 46; do
  fleche run infer_gen_graph --env DATASET=novelty \
    --env CONFIG=llama_gen_graph_seed${seed}
done

# 5.3 Qwen basic (seeds 42-46)
for seed in 42 43 44 45 46; do
  fleche run infer_gen --env DATASET=novelty \
    --env CONFIG=qwen_gen_basic_orc_seed${seed}
done

# 5.4 Qwen graph (seeds 43-46)
for seed in 43 44 45 46; do
  fleche run infer_gen_graph --env DATASET=novelty \
    --env CONFIG=qwen3_32b_graph_a100_lr1e4_seed${seed}
done
```

**Note**: The exact fleche env vars may need adjusting to match the model directory
naming conventions.

## Phase 6: CSAbstruct evaluation

Evaluate our LLM-based abstract background/target classification against CSAbstruct
gold labels. Data: 295 dev entries (already downloaded to `output/csabstruct_dev.json`).

### Steps

1. **Run LLM classification**: For each of the 295 abstracts, call the abstract
   classification pipeline (`annotate_paper.py`) to produce predicted `background`
   and `target` text blocks.

2. **Compute text-overlap metrics**: Compare predicted vs gold text blocks using:
   - ROUGE-1, ROUGE-2, ROUGE-L (n-gram overlap)
   - BERTScore (semantic similarity)

3. **Compute sentence-level metrics**: Split predicted and gold text blocks back
   into sentences, then compute:
   - Exact sentence match accuracy (% of sentences in the correct bucket)
   - Sentence-level F1 (precision/recall for background vs target classification)

### Notes

- Gold labels map CSAbstruct categories to binary: `background` stays as-is,
  everything else (`objective`, `method`, `result`, `other`) becomes `target`.
- Our classifier outputs two continuous text blocks, not sentence-level labels,
  so sentence matching requires re-segmentation.

## Estimated costs

- GPT-4o-mini experiments: ~$2-4/run x 5 runs per config
- Gemini experiments: comparable or lower
- Finetuning: GPU time (depends on hardware)
- NovaSCORE/SciMON: minimal API cost (local compute + embedding calls)
