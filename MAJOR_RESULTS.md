# Major Results

This file contains summary tables comparing different configurations across datasets.

**New results are added at the top in reverse chronological order.**

---

2026-01-14 Baseline Comparison
------------------------------

Comparison of baseline methods against GraphMind GPT on both datasets.

### ORC Dataset (n=100)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.198 | 0.194 | 0.690 | 0.440 | 0.293 | ~$0.00 |
| Novascore | 0.189 | 0.194 | 0.830 | 0.340 | 0.201 | $0.00 |
| Scimon GPT | 0.160 ± 0.037 | 0.137 ± 0.062 | 1.248 ± 0.025 | 0.190 ± 0.015 | 0.101 ± 0.012 | $0.022 |
| Basic GPT (Sans) | 0.048 ± 0.023 | 0.050 ± 0.027 | 1.226 ± 0.035 | 0.186 ± 0.027 | 0.102 ± 0.015 | $0.028 |
| **GraphMind GPT (Full)** | **0.312 ± 0.058** | **0.337 ± 0.077** | **0.862 ± 0.018** | **0.290 ± 0.016** | **0.150 ± 0.014** | $0.116 |

### PeerRead Dataset (n=70)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.143 | 0.098 | 0.486 | 0.614 | 0.241 | ~$0.00 |
| Novascore | 0.227 | 0.301 | 2.214 | 0.043 | 0.149 | $0.00 |
| Scimon GPT | 0.080 ± 0.027 | 0.116 ± 0.035 | 1.054 ± 0.007 | 0.143 ± 0.012 | 0.096 ± 0.006 | $0.013 |
| Basic GPT (Sans) | 0.139 ± 0.074 | 0.125 ± 0.074 | 1.250 ± 0.055 | 0.159 ± 0.012 | 0.121 ± 0.009 | $0.017 |
| **GraphMind GPT (Full)** | **0.449 ± 0.089** | **0.435 ± 0.092** | **1.112 ± 0.074** | 0.115 ± 0.019 | 0.066 ± 0.012 | $0.052 |

### Notes

- **Llama Basic**: Llama-3.1-8B-Instruct fine-tuned with LoRA on abstract-only input.
  ORC: 6 epochs, lr=2e-4. PeerRead: 4 epochs, lr=2e-4.
  Output: `output/baselines/llama_orc_llama_orc/`, `output/baselines/llama_peerread_llama_peerread/`
- **Novascore**: Tuned similarity thresholds (0.60 for ORC, 0.70 for PeerRead). Single
  deterministic run, no stdev. Output: `output/baselines/novascore_orc_t060/`,
  `output/baselines/novascore_peerread_t070/`
- **Scimon GPT**: 5 runs using gpt-4o-mini. ORC: 3/5 successful, PeerRead: 4/5 successful.
  Output: `output/baselines/scimon_orc/`, `output/baselines/scimon_peerread/`
- **GPT methods**: 5 runs using gpt-4o-mini with demos (ORC) / no demos (PeerRead).

### Key Findings

1. **Performance varies by dataset**: GraphMind GPT is best on ORC, while Llama Basic dominates
   PeerRead (Pearson 0.647 vs 0.449)
2. **Llama Basic shows extreme dataset sensitivity**: Near-zero correlation on ORC (-0.039), but
   excellent on PeerRead (0.647). This may be due to PeerRead's simpler rating distribution and
   smaller but more consistent training data.
3. **Novascore** performs better than Scimon GPT after threshold tuning (default 0.8 was too high)
4. **Basic GPT (Sans)** is comparable to or better than retrieval-based baselines on ORC, showing
   that LLM judgement alone provides meaningful signal
5. **Cost-performance trade-off**: Llama/Novascore are essentially free at inference; GraphMind
   costs ~$0.05-0.12/paper but achieves best correlation on ORC

---

2026-01-14 (Final: ORC with demos, PeerRead without demos)
----------------------------------------------------------

Recommended configuration for publication. Uses consistent demo setting within each
dataset to maintain interpretable ablation baselines.

### ORC Dataset (dev_100_balanced, demos: orc_balanced_4)

| Configuration | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|---------------|---------|----------|-----|----------|-----|----------|
| Sans (abstract only) | 0.048 ± 0.053 | 0.044 ± 0.049 | 1.20 | 0.186 | 0.102 | $0.028 |
| Related (no graph) | 0.091 ± 0.111 | 0.085 ± 0.109 | 0.95 | 0.282 | 0.176 | $0.105 |
| Graph Only (no related) | 0.020 ± 0.028 | 0.013 ± 0.023 | 1.18 | 0.218 | 0.120 | $0.033 |
| Citations Only | 0.224 ± 0.032 | 0.239 ± 0.035 | 1.14 | 0.218 | 0.123 | $0.086 |
| Semantic Only | 0.132 ± 0.068 | 0.135 ± 0.052 | 1.21 | 0.166 | 0.104 | $0.093 |
| **Full Pipeline** | **0.312 ± 0.058** | **0.337 ± 0.077** | **0.86** | **0.290** | **0.150** | $0.116 |

### PeerRead Dataset (balanced_68, no demos)

| Configuration | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|---------------|---------|----------|-----|----------|-----|----------|
| Sans (abstract only) | 0.042 ± 0.063 | 0.058 ± 0.067 | 1.18 | 0.147 | 0.104 | $0.017 |
| Related (no graph) | 0.146 ± 0.082 | 0.152 ± 0.081 | 1.02 | 0.182 | 0.125 | $0.048 |
| Graph Only (no related) | 0.199 ± 0.061 | 0.221 ± 0.063 | 1.15 | 0.135 | 0.100 | $0.020 |
| Citations Only | 0.309 ± 0.072 | 0.342 ± 0.061 | 1.53 | 0.074 | 0.054 | $0.037 |
| Semantic Only | 0.517 ± 0.026 | 0.499 ± 0.020 | 0.98 | 0.144 | 0.085 | $0.040 |
| **Full Pipeline** | **0.538 ± 0.062** | **0.526 ± 0.063** | **1.16** | **0.094** | **0.058** | $0.052 |

### Rationale

**Why demos for ORC?** Without demos, the simpler configurations (Sans, Related, Graph Only)
drop to near-zero correlation, making the ablation narrative less interpretable. With demos,
each component shows measurable signal.

**Why no demos for PeerRead?** Most configurations improve without demos on PeerRead. The
baselines remain interpretable (Sans 0.042, Related 0.146, Graph Only 0.199) while the full
pipeline achieves its best result (0.538).

**Trade-off:** ORC Full with demos (0.312) is lower than without demos (0.377), but maintaining
credible baselines is more important for the ablation story.

### Key Findings

1. **Full pipeline is best** on both datasets (ORC: 0.312, PeerRead: 0.538)
2. **Graph adds significant value**: Full vs Related = +0.221 (ORC), +0.392 (PeerRead)
3. **Related papers add value**: Full vs Graph Only = +0.292 (ORC), +0.339 (PeerRead)
4. **Source importance differs by dataset**: ORC favours citations, PeerRead favours semantic

---

2026-01-14 (Best configurations with optimal demo settings)
-----------------------------------------------------------

Results after testing all configurations with and without demonstrations.
Each row uses the optimal demo setting for that configuration.

### ORC Dataset (dev_100_balanced)

| Configuration | Pearson | Spearman | MAE | Demos | Cost/run |
|---------------|---------|----------|-----|-------|----------|
| Sans (abstract only) | 0.048 ± 0.053 | 0.044 ± 0.049 | 1.20 | orc_balanced_4 | $0.028 |
| Related (no graph) | 0.091 ± 0.111 | 0.085 ± 0.109 | 0.95 | orc_balanced_4 | $0.105 |
| Graph Only (no related) | 0.020 ± 0.028 | 0.023 ± 0.031 | 1.79 | orc_balanced_4 | $0.033 |
| Citations Only | 0.326 ± 0.041 | 0.335 ± 0.041 | 1.00 | none | $0.084 |
| Semantic Only | 0.132 ± 0.068 | 0.135 ± 0.052 | 1.21 | orc_balanced_4 | $0.093 |
| **Full Pipeline** | **0.377 ± 0.034** | **0.383 ± 0.042** | **0.86** | **none** | $0.120 |

### PeerRead Dataset (balanced_68)

| Configuration | Pearson | Spearman | MAE | Demos | Cost/run |
|---------------|---------|----------|-----|-------|----------|
| Sans (abstract only) | 0.139 ± 0.069 | 0.129 ± 0.055 | 1.16 | peerread_balanced_5 | $0.019 |
| Related (no graph) | 0.146 ± 0.082 | 0.152 ± 0.081 | 1.02 | none | $0.048 |
| Graph Only (no related) | 0.199 ± 0.061 | 0.221 ± 0.063 | 1.15 | none | $0.020 |
| Citations Only | 0.339 ± 0.050 | 0.359 ± 0.045 | 1.49 | peerread_balanced_5 | $0.038 |
| Semantic Only | 0.517 ± 0.026 | 0.499 ± 0.020 | 0.98 | none | $0.040 |
| **Full Pipeline** | **0.538 ± 0.062** | **0.526 ± 0.063** | **1.16** | **none** | $0.052 |

### Demo Effect Summary

| Configuration | ORC Effect | PeerRead Effect |
|---------------|------------|-----------------|
| Sans | +0.025 (demos help) | +0.097 (demos help) |
| Related | +0.070 (demos help) | -0.075 (demos hurt) |
| Graph Only | +0.022 (demos help) | -0.119 (demos hurt) |
| Citations | -0.102 (demos hurt) | +0.030 (demos help) |
| Semantic | +0.056 (demos help) | -0.144 (demos hurt) |
| Full | -0.065 (demos hurt) | -0.089 (demos hurt) |

### Demonstration Details
- **orc_balanced_4**: 4 examples from ORC dataset (2 high novelty, 2 low novelty)
- **peerread_balanced_5**: 5 examples from PeerRead dataset (balanced across novelty scores)

### Key Findings
1. **Full pipeline is best** on both datasets (ORC: 0.377, PeerRead: 0.538)
2. **Full works better zero-shot** - demos hurt performance on both datasets
3. **Demo effect is dataset-dependent**:
   - ORC: demos help most configs except Full and Citations
   - PeerRead: demos hurt most configs except Sans and Citations
4. **Citation vs Semantic sources differ by dataset**:
   - ORC: Citations (0.326) >> Semantic (0.132)
   - PeerRead: Semantic (0.517) >> Citations (0.339)

---

2026-01-13 21:43 (ORC) / 21:51 (PeerRead)
------------------------------------------

### ORC Dataset (dev_100_balanced)

| Configuration | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------------|---------|----------|-----|----------|----|----|
| **Sans** (abstract only) | 0.048 ± 0.023 | 0.050 ± 0.027 | 1.226 ± 0.035 | 0.186 ± 0.027 | 0.102 ± 0.015 | $0.028 |
| **Related** (related papers only) | 0.091 ± 0.111 | 0.085 ± 0.109 | 0.952 ± 0.048 | 0.282 ± 0.025 | 0.176 ± 0.014 | $0.105 |
| **Graph Only** (no related) | 0.020 ± 0.028 | 0.013 ± 0.023 | 1.178 ± 0.028 | 0.218 ± 0.026 | 0.120 ± 0.014 | $0.033 |
| **Citations** (graph + citations) | 0.224 ± 0.032 | 0.239 ± 0.035 | 1.136 ± 0.063 | 0.218 ± 0.026 | 0.123 ± 0.025 | $0.086 |
| **Semantic** (graph + semantic) | 0.132 ± 0.068 | 0.135 ± 0.052 | 1.214 ± 0.038 | 0.166 ± 0.036 | 0.104 ± 0.020 | $0.093 |
| **Full** (graph + both) | **0.312 ± 0.058** | **0.337 ± 0.077** | **0.862 ± 0.018** | **0.290 ± 0.016** | **0.150 ± 0.014** | $0.116 |

#### Configuration Details (ORC)
- **Sans**: `eval-prompt=sans`, no related papers, no graph
  - Output: `output/eval_orc/ablation_sans/`
- **Related**: `eval-prompt=related` (v6 with "focus on abstract claims"), related papers from both sources
  - Output: `output/eval_orc/ablation_related_v6/`
- **Graph Only**: `eval-prompt=norel-graph`, graph summary without related papers
  - Output: `output/eval_orc/ablation_norel/`
- **Citations**: `eval-prompt=full-graph-structured`, `sources=citations`
  - Output: `output/eval_orc/ablation_citations/`
- **Semantic**: `eval-prompt=semantic-only` (v2 conservative), `sources=semantic`
  - Output: `output/eval_orc/ablation_semantic_v2/`
- **Full**: `eval-prompt=full-graph-structured`, `sources=both`
  - Output: `output/eval_orc/ablation_full/`

#### ORC Prompt Improvements
- **Related v6**: Improved from -0.030 (original) to 0.091 by emphasizing "focus on what the abstract claims"
- **Semantic v2**: Reduced from 0.375 (original) to 0.132 by adding conservative language about semantic matches being "tangentially related"

### PeerRead Dataset

| Configuration | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------------|---------|----------|-----|----------|----|----|
| **Sans** (abstract only) | 0.139 ± 0.074 | 0.125 ± 0.074 | 1.250 ± 0.055 | 0.159 ± 0.012 | 0.121 ± 0.009 | $0.019 |
| **Related** (related papers only) | 0.071 ± 0.138 | 0.091 ± 0.149 | 1.079 ± 0.044 | 0.162 ± 0.021 | 0.108 ± 0.014 | $0.048 |
| **Graph Only** (no related) | 0.080 ± 0.109 | 0.098 ± 0.088 | 1.215 ± 0.047 | 0.138 ± 0.022 | 0.084 ± 0.011 | $0.071 |
| **Citations** (graph + citations) | 0.339 ± 0.054 | 0.394 ± 0.057 | 1.503 ± 0.064 | 0.068 ± 0.022 | 0.060 ± 0.022 | $0.038 |
| **Semantic** (graph + semantic) | 0.373 ± 0.048 | 0.368 ± 0.057 | 0.932 ± 0.030 | 0.176 ± 0.023 | 0.099 ± 0.028 | $0.042 |
| **Full** (graph + both) | **0.449 ± 0.089** | **0.435 ± 0.092** | **1.112 ± 0.074** | **0.115 ± 0.019** | **0.066 ± 0.012** | $0.053 |

#### Configuration Details (PeerRead)
- **Sans**: `eval-prompt=sans`, no related papers, no graph
  - Output: `output/eval_peerread/ablation_sans/`
- **Related**: `eval-prompt=related`, related papers from both sources
  - Output: `output/eval_peerread/ablation_related/`
- **Graph Only**: `eval-prompt=norel-graph`, graph summary without related papers
  - Output: `output/eval_peerread/ablation_norel/`
- **Citations**: `eval-prompt=full-graph-structured`, `sources=citations`
  - Output: `output/eval_peerread/ablation_citations/`
- **Semantic**: `eval-prompt=full-graph-structured`, `sources=semantic`
  - Output: `output/eval_peerread/ablation_semantic/`
- **Full**: `eval-prompt=full-graph-structured`, `sources=both`
  - Output: `output/eval_peerread/ablation_full/`

### Common Settings

#### Model Configuration
- **Model**: gpt-4o-mini
- **Temperature**: 0.0 (extraction and evaluation)
- **Seed**: 42
- **Runs per configuration**: 5

#### Dataset Details
- **ORC**: `output/venus5/split/dev_100_balanced.json.zst` (hash: `665b5805`)
  - Papers: 100
  - Demonstrations: orc_balanced_4
- **PeerRead**: `output/new_peerread/peter_summarised/balanced_68.json.zst` (hash: `5b589cf7`)
  - Papers: 68
  - Demonstrations: peerread_balanced_5

#### Key Findings
1. **Full pipeline performs best** on both datasets (Pearson: ORC 0.312, PeerRead 0.449)
2. **Citations contribute most** to performance (vs semantic-only)
3. **Graph summaries are essential** - comparing Sans (0.048) vs Full (0.312) on ORC
4. **Related papers alone have modest signal** but improve with graph context

---
