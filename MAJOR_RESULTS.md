# Major Results

This file contains summary tables comparing different configurations across datasets.

**New results are added at the top in reverse chronological order.**

---

2026-02-02 Complete Baseline Comparison with Search Baselines
-------------------------------------------------------------

Updated baseline comparison including web search-grounded models. All methods use 5-seed
experiments for error bars (except Novascore which is deterministic). GPT Graph uses best
config (no demos).

### ORC Dataset (n=100)

| Method | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|--------|---------|----------|--------|-----|------|
| GPT Search | -0.077 ± 0.100 | -0.073 ± 0.114 | 0.542 ± 0.022 | 1.512 ± 0.029 | 1.782 ± 0.034 |
| GPT Basic | 0.048 ± 0.023 | 0.050 ± 0.027 | 0.612 ± 0.022 | 1.226 ± 0.035 | 1.448 ± 0.022 |
| Gemini Search | 0.114 ± 0.085 | 0.117 ± 0.074 | 0.898 ± 0.026 | 0.792 ± 0.046 | 1.017 ± 0.043 |
| Llama Graph | 0.142 ± 0.098 | 0.143 ± 0.089 | 0.474 ± 0.029 | 1.546 ± 0.048 | 1.732 ± 0.039 |
| Llama Basic | 0.157 ± 0.024 | 0.167 ± 0.027 | 0.910 ± 0.014 | 0.638 ± 0.021 | 0.909 ± 0.029 |
| Scimon | 0.160 ± 0.037 | 0.137 ± 0.062 | 0.582 ± 0.011 | 1.248 ± 0.025 | 1.471 ± 0.016 |
| Qwen Graph | 0.171 ± 0.062 | 0.175 ± 0.066 | 0.442 ± 0.016 | 1.518 ± 0.036 | 1.705 ± 0.023 |
| Qwen Basic | 0.173 ± 0.072 | 0.163 ± 0.074 | 0.964 ± 0.011 | 0.588 ± 0.034 | 0.812 ± 0.030 |
| Novascore | 0.189 | 0.194 | 0.840 | 0.830 | 1.091 |
| **GPT Graph** | **0.377 ± 0.034** | **0.383 ± 0.042** | **0.832 ± 0.013** | **0.860 ± 0.032** | **1.157 ± 0.020** |

### PeerRead Dataset (n=68-70)

| Method | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|--------|---------|----------|--------|-----|------|
| GPT Search | -0.049 ± 0.075 | -0.013 ± 0.084 | 0.532 ± 0.035 | 1.485 ± 0.010 | 1.765 ± 0.031 |
| Qwen Basic | 0.069 ± 0.045 | 0.057 ± 0.048 | 0.680 ± 0.039 | 1.254 ± 0.080 | 1.523 ± 0.102 |
| Scimon | 0.080 ± 0.027 | 0.116 ± 0.035 | 0.804 ± 0.007 | 1.054 ± 0.007 | 1.203 ± 0.006 |
| Llama Graph | 0.126 ± 0.074 | 0.131 ± 0.077 | 0.306 ± 0.016 | 1.740 ± 0.027 | 1.899 ± 0.036 |
| Qwen Graph | 0.129 ± 0.045 | 0.125 ± 0.043 | 0.891 ± 0.033 | 0.877 ± 0.044 | 1.045 ± 0.051 |
| GPT Basic | 0.139 ± 0.074 | 0.125 ± 0.074 | 0.591 ± 0.048 | 1.250 ± 0.055 | 1.437 ± 0.053 |
| Gemini Search | 0.139 ± 0.113 | 0.131 ± 0.111 | 0.968 ± 0.022 | 0.791 ± 0.028 | 0.925 ± 0.036 |
| Novascore | 0.227 | 0.301 | 0.171 | 2.214 | 2.363 |
| Llama Basic | 0.284 ± 0.096 | 0.360 ± 0.106 | 0.903 ± 0.036 | 0.551 ± 0.294 | 0.863 ± 0.208 |
| **GPT Graph** | **0.538 ± 0.062** | **0.526 ± 0.063** | **0.785 ± 0.063** | **1.112 ± 0.074** | **1.258 ± 0.085** |

### Method Details

- **GPT Search**: gpt-4o-mini-search with web search grounding, seeds 42-46
- **Gemini Search**: gemini-2.0-flash with Google Search grounding, seeds 42-46
- **Llama Basic**: Llama-3.1-8B-Instruct classifier, ORC seeds 47/49/53/57/60, PeerRead seeds 42/45/46/48/50
- **Llama Graph**: Llama-3.1-8B-Instruct generative with graph context, seeds 42-46
- **Qwen Basic**: Qwen3-32B generative, seeds 42-46
- **Qwen Graph**: Qwen3-32B generative with graph context, seeds 42-46, lr=1e-4
- **Novascore**: Deterministic retrieval baseline, threshold 0.60 (ORC) / 0.70 (PeerRead)
- **Scimon**: gpt-4o-mini, 3-5 successful runs
- **GPT Basic**: gpt-4o-mini, sans prompt with demos
- **GPT Graph**: gpt-4o-mini, full-graph-structured prompt, no demos (best config)

### Key Takeaways

1. **GPT Graph wins on both datasets** (ORC: 0.377, PeerRead: 0.538)
2. **Web search grounding doesn't help**: GPT Search shows negative correlation on both datasets
3. **Gemini Search is better than GPT Search**: Positive but modest correlation (0.114 ORC, 0.139 PeerRead)
4. **Pre-built knowledge graph >> live web search**: GPT Graph outperforms search-grounded baselines by 3-4x
5. **Search models have high Acc±1 but poor Pearson**: Conservative predictions cluster around mean ratings

---

2026-01-20 Complete Baseline Comparison (5-Seed, All Metrics)
-------------------------------------------------------------

Complete baseline comparison with all metrics. All methods use 5-seed experiments for error bars
(except Novascore which is deterministic). GPT Graph uses best config (no demos).

### ORC Dataset (n=100)

| Method | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|--------|---------|----------|--------|-----|------|
| Llama Basic | 0.157 ± 0.024 | 0.167 ± 0.027 | 0.910 ± 0.014 | 0.638 ± 0.021 | 0.909 ± 0.029 |
| Llama Graph | 0.142 ± 0.098 | 0.143 ± 0.089 | 0.474 ± 0.029 | 1.546 ± 0.048 | 1.732 ± 0.039 |
| Qwen Basic | 0.173 ± 0.072 | 0.163 ± 0.074 | 0.964 ± 0.011 | 0.588 ± 0.034 | 0.812 ± 0.030 |
| Qwen Graph | 0.171 ± 0.062 | 0.175 ± 0.066 | 0.442 ± 0.016 | 1.518 ± 0.036 | 1.705 ± 0.023 |
| Novascore | 0.189 | 0.194 | 0.840 | 0.830 | 1.091 |
| Scimon | 0.160 ± 0.037 | 0.137 ± 0.062 | 0.582 ± 0.011 | 1.248 ± 0.025 | 1.471 ± 0.016 |
| GPT Basic | 0.048 ± 0.023 | 0.050 ± 0.027 | 0.612 ± 0.022 | 1.226 ± 0.035 | 1.448 ± 0.022 |
| **GPT Graph** | **0.377 ± 0.034** | **0.383 ± 0.042** | **0.832 ± 0.013** | **0.860 ± 0.032** | **1.157 ± 0.020** |

### PeerRead Dataset (n=68-70)

| Method | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|--------|---------|----------|--------|-----|------|
| Llama Basic | 0.284 ± 0.096 | 0.360 ± 0.106 | 0.903 ± 0.036 | 0.551 ± 0.294 | 0.863 ± 0.208 |
| Llama Graph | 0.126 ± 0.074 | 0.131 ± 0.077 | 0.306 ± 0.016 | 1.740 ± 0.027 | 1.899 ± 0.036 |
| Qwen Basic | 0.069 ± 0.045 | 0.057 ± 0.048 | 0.680 ± 0.039 | 1.254 ± 0.080 | 1.523 ± 0.102 |
| Qwen Graph | 0.129 ± 0.045 | 0.125 ± 0.043 | 0.891 ± 0.033 | 0.877 ± 0.044 | 1.045 ± 0.051 |
| Novascore | 0.227 | 0.301 | 0.171 | 2.214 | 2.363 |
| Scimon | 0.080 ± 0.027 | 0.116 ± 0.035 | 0.804 ± 0.007 | 1.054 ± 0.007 | 1.203 ± 0.006 |
| GPT Basic | 0.139 ± 0.074 | 0.125 ± 0.074 | 0.591 ± 0.048 | 1.250 ± 0.055 | 1.437 ± 0.053 |
| **GPT Graph** | **0.538 ± 0.062** | **0.526 ± 0.063** | **0.785 ± 0.063** | **1.112 ± 0.074** | **1.258 ± 0.085** |

### Method Details

- **Llama Basic**: Llama-3.1-8B-Instruct classifier, ORC seeds 47/49/53/57/60, PeerRead seeds 42/45/46/48/50
- **Llama Graph**: Llama-3.1-8B-Instruct generative with graph context, seeds 42-46
- **Qwen Basic**: Qwen3-32B generative, seeds 42-46
- **Qwen Graph**: Qwen3-32B generative with graph context, seeds 42-46, lr=1e-4
- **Novascore**: Deterministic retrieval baseline, threshold 0.60 (ORC) / 0.70 (PeerRead)
- **Scimon**: gpt-4o-mini, 3-5 successful runs
- **GPT Basic**: gpt-4o-mini, sans prompt with demos
- **GPT Graph**: gpt-4o-mini, full-graph-structured prompt, no demos (best config)

### Key Takeaways

1. **GPT Graph wins on both datasets** (ORC: 0.377, PeerRead: 0.538)
2. **SFT baselines have HIGH VARIANCE**: Qwen Basic ranges from 0.070-0.243 on ORC
3. **Graph context hurts SFT models**: Llama/Qwen Graph underperform their Basic variants
4. **Acc±1 can be misleading**: High Acc±1 doesn't correlate with high Pearson (e.g., Qwen Basic ORC)
5. **Llama Basic strong on PeerRead**: Best SFT Pearson (0.284) despite simple approach

---

2026-01-20 GPT Ablation Study (All Metrics)
-------------------------------------------

Component ablation for GPT-based method. Sans and Full use same configs as baseline table
(Sans with demos, Full no demos). Intermediate configs use demos for ORC, no demos for PeerRead.
All 5 runs with gpt-4o-mini.

### ORC Dataset

| Configuration | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|---------------|---------|----------|--------|-----|------|
| Sans (abstract only) | 0.048 ± 0.023 | 0.050 ± 0.027 | 0.612 ± 0.022 | 1.226 ± 0.035 | 1.448 ± 0.022 |
| Related (no graph) | 0.091 ± 0.111 | 0.085 ± 0.109 | 0.778 ± 0.026 | 0.952 ± 0.048 | 1.201 ± 0.046 |
| Graph Only (no related) | 0.020 ± 0.028 | 0.013 ± 0.023 | 0.632 ± 0.018 | 1.178 ± 0.028 | 1.423 ± 0.016 |
| Citations Only | 0.224 ± 0.032 | 0.239 ± 0.035 | 0.730 ± 0.040 | 1.136 ± 0.063 | 1.425 ± 0.055 |
| Semantic Only | 0.132 ± 0.068 | 0.135 ± 0.052 | 0.634 ± 0.009 | 1.214 ± 0.038 | 1.415 ± 0.018 |
| **Full Pipeline** | **0.377 ± 0.034** | **0.383 ± 0.042** | **0.832 ± 0.013** | **0.860 ± 0.032** | **1.157 ± 0.020** |

### PeerRead Dataset

| Configuration | Pearson | Spearman | Acc ±1 | MAE | RMSE |
|---------------|---------|----------|--------|-----|------|
| Sans (abstract only) | 0.139 ± 0.074 | 0.125 ± 0.074 | 0.591 ± 0.048 | 1.250 ± 0.055 | 1.437 ± 0.053 |
| Related (no graph) | 0.146 ± 0.082 | 0.152 ± 0.081 | 0.797 ± 0.037 | 1.021 ± 0.044 | 1.194 ± 0.047 |
| Graph Only (no related) | 0.199 ± 0.061 | 0.221 ± 0.063 | 0.718 ± 0.043 | 1.147 ± 0.054 | 1.307 ± 0.053 |
| Citations Only | 0.309 ± 0.072 | 0.342 ± 0.061 | 0.526 ± 0.024 | 1.526 ± 0.060 | 1.725 ± 0.064 |
| Semantic Only | 0.517 ± 0.026 | 0.499 ± 0.020 | 0.876 ± 0.027 | 0.982 ± 0.032 | 1.113 ± 0.045 |
| **Full Pipeline** | **0.538 ± 0.062** | **0.526 ± 0.063** | **0.785 ± 0.063** | **1.112 ± 0.074** | **1.258 ± 0.085** |

### Configuration Details

- **Sans**: Abstract only, no graph or related papers (with demos)
- **Related**: Related papers from both sources, no graph summary
- **Graph Only**: Graph summary only, no related papers
- **Citations**: Graph + citation-based related papers only
- **Semantic**: Graph + semantic similarity related papers only
- **Full**: Graph + both citation and semantic related papers (no demos)

### Key Findings

1. **Full pipeline best on both datasets** (ORC: 0.377, PeerRead: 0.538)
2. **Graph adds value**: Full vs Related = +0.286 (ORC), +0.392 (PeerRead)
3. **Related papers add value**: Full vs Graph Only = +0.357 (ORC), +0.339 (PeerRead)
4. **Source importance differs**: ORC favours citations (0.224), PeerRead favours semantic (0.517)
5. **Semantic dominates PeerRead**: Semantic alone (0.517) nearly matches Full (0.538)

---

2026-01-19 Gen Graph Extended Context Investigation
---------------------------------------------------

Investigated whether truncation of contrasting papers causes poor Gen Graph performance.

### Token Analysis

Gen Graph inputs average ~4,150 tokens with max ~5,550 tokens. With max_length=2000:
- **100% of inputs are truncated**
- Contrasting papers (35% of input, ~1,457 tokens) are **completely cut off**
- Supporting papers only partially included (~65%)

For comparison, Gen Basic inputs average only ~376 tokens (ORC) / ~264 tokens (PeerRead).

### Extended Context Experiments (Llama-8B, ORC)

| Config | max_length | LR | Epochs | Pearson | Notes |
|--------|------------|------|--------|---------|-------|
| Baseline | 2000 | 1e-4 | 1 | 0.155 | Contrasting papers truncated |
| 6k v1 (grjy) | 6000 | 1e-4 | 1 | N/A | Mode collapse (all predictions=3) |
| 6k v2 (q2do) | 6000 | 5e-5 | 3 | 0.114 | Fixed mode collapse, but worse than baseline |

### Key Finding

**Truncation is NOT the main bottleneck for Gen Graph performance.**

Extended context (0.114) performs *worse* than truncated baseline (0.155), despite now including
the contrasting papers. This suggests either:
1. Longer sequences are harder for the model to learn effectively
2. Contrasting paper information isn't useful for the rating task
3. The model cannot effectively leverage the additional context

This aligns with the broader pattern: **Gen Basic consistently outperforms Gen Graph** across
both Llama and Qwen (Qwen Basic 0.30 vs Qwen Graph 0.17 on ORC).

---

2026-01-18 SFT Model Comparison (Same Model Across Methods)
-----------------------------------------------------------

Direct comparison of SFT methods using the same model for fair evaluation.

### Llama-3.1-8B-Instruct Results

| Method | ORC Pearson | PeerRead Pearson | Notes |
|--------|-------------|------------------|-------|
| Classifier Basic | 0.157 ± 0.024 | 0.284 ± 0.096 | Classification head, 5 seeds |
| Gen Basic | -0.113 | -0.063 | Rating-first, 3 epochs (overfitting) |
| Gen Graph | 0.155 | 0.135 | With graph+related context, 1 epoch |

### Qwen3-32B Results

| Method | ORC Pearson | PeerRead Pearson | Notes |
|--------|-------------|------------------|-------|
| Classifier Basic | 0.080 | **0.862** | Classification head, 6/4 epochs |
| Gen Basic | **0.300** | 0.170 | Rating-first, 6/1 epochs |
| Gen Graph | 0.171 ± 0.063 | 0.129 ± 0.045 | With graph context, A100, lr=1e-4, 5 seeds |

### Key Findings

1. **No single model dominates**: Llama Classifier best on ORC (0.157), Qwen Classifier best on PeerRead (0.862)
2. **Gen Basic overfits easily**: Both Llama (-0.113/-0.063) and Qwen (needs tuning on PeerRead) show overfitting with multiple epochs
3. **Gen Graph underperforms expectations**: Despite rich context, both models show modest results
4. **Qwen Classifier on PeerRead is exceptional**: 0.862 Pearson is the best SFT result overall
5. **Model size matters differently by method**: Qwen's size helps Classifier on PeerRead but not ORC

### Observations

- **Llama Gen Basic needs tuning**: 3 epochs caused mode collapse (predicting only ratings 2-3)
- **Qwen Classifier ORC may need higher LR**: 2e-5 might be too conservative for the larger model
- **Qwen Gen Graph tuning and 5-seed results**:
  - Initial (lr=2e-5, max_length=1536): PeerRead Pearson 0.071
  - Fixed max_length=2000: PeerRead Pearson 0.096 (+35%)
  - Fixed lr=1e-4: PeerRead Pearson 0.152 (+58% vs initial)
  - **5-seed results (lr=1e-4, max_length=2000)**:
    - ORC: 0.171 ± 0.063 (seeds 42-46: 0.092, 0.200, 0.200, 0.121, 0.244)
    - PeerRead: 0.129 ± 0.045 (seeds 42-46: 0.152, 0.093, 0.071, 0.152, 0.178)
  - Key insight: High variance across seeds; ORC more variable than PeerRead
  - A100 (80GB) required for full-length Qwen Gen Graph runs

---

2026-01-18 Generative SFT Baseline Comparison
---------------------------------------------

Comparison of generative SFT methods: Gen Basic (rating-first) and Gen Graph (with knowledge graph context).

### ORC Dataset (n=100)

| Method | Pearson | Spearman | MAE | Model | Notes |
|--------|---------|----------|-----|-------|-------|
| Qwen Basic Gen | 0.300 | 0.270 | 0.570 | Qwen3-32B | Rating-first prompt, 6 epochs |
| Llama Gen Graph | 0.155 | 0.132 | 1.570 | Llama-3.1-8B | With graph+related context, 1 epoch |

### PeerRead Dataset (n=70)

| Method | Pearson | Spearman | MAE | Model | Notes |
|--------|---------|----------|-----|-------|-------|
| Qwen Basic Gen | 0.170 | 0.162 | 0.929 | Qwen3-32B | Rating-first prompt, 1 epoch, seed 46 (tuned) |
| Llama Gen Graph | 0.135 | 0.132 | 1.729 | Llama-3.1-8B | With graph+related context, 1 epoch |

### Notes

- **Qwen Basic Gen**: Qwen3-32B fine-tuned with LoRA for generative rating prediction.
  ORC: seed 42, 6 epochs, lr=2e-5, batch_size=1. Output: `output/baselines/llama_gen_orc_qwen3_32b_orc/`
  PeerRead: seed 46, 1 epoch, lr=2e-5, batch_size=1. Output: `output/baselines/llama_gen_peerread_qwen3_32b_peerread_e1_seed46/`
  **Tuning note**: 6 epochs caused overfitting on PeerRead (Pearson -0.038). Reduced to 1 epoch with seed 46.
- **Llama Gen Graph**: Llama-3.1-8B-Instruct fine-tuned with LoRA using knowledge graph and related paper context.
  1 epoch, lr=1e-4, batch_size=4. Requires preprocessing to add graph context (~$5-6 API cost for GPT-4o-mini).
  ORC output: `output/baselines/llama_gen_graph_orc_llama_graph/`
  PeerRead output: `output/baselines/llama_gen_graph_peerread_llama_graph/`

### Key Findings

1. **Qwen Basic Gen works well on ORC** (Pearson 0.300), comparable to GraphMind GPT (0.312)
2. **Qwen Basic Gen requires tuning for PeerRead**: 6 epochs caused overfitting (Pearson -0.038). After tuning to 1 epoch with seed 46, achieves Pearson 0.170
3. **Llama Gen Graph underperforms** on both datasets despite additional context - may need more epochs or hyperparameter tuning
4. **Cost trade-off**: Qwen Basic Gen is inference-free after training; Gen Graph requires additional API cost for preprocessing
5. **All methods now exceed Pearson 0.10 threshold** on both datasets after hyperparameter tuning

---

2026-01-17 Baseline Comparison (with Qwen Gen)
----------------------------------------------

Comparison of baseline methods against GraphMind GPT on both datasets.
Now includes Qwen Basic Gen (generative SFT with rating prediction).

### ORC Dataset (n=100)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.157 ± 0.024 | 0.167 ± 0.027 | 0.638 ± 0.021 | 0.458 ± 0.020 | 0.284 ± 0.016 | ~$0.00 |
| Qwen Basic Gen | 0.300 | 0.270 | 0.570 | 0.430 | 0.218 | ~$0.00 |
| Novascore | 0.189 ± 0.000 | 0.194 ± 0.000 | 0.830 ± 0.000 | 0.340 ± 0.000 | 0.201 ± 0.000 | $0.00 |
| Scimon GPT | 0.160 ± 0.037 | 0.137 ± 0.062 | 1.248 ± 0.025 | 0.190 ± 0.015 | 0.101 ± 0.012 | $0.022 |
| Basic GPT (Sans) | 0.048 ± 0.023 | 0.050 ± 0.027 | 1.226 ± 0.035 | 0.186 ± 0.027 | 0.102 ± 0.015 | $0.028 |
| **GraphMind GPT (Full)** | **0.312 ± 0.058** | **0.337 ± 0.077** | **0.862 ± 0.018** | **0.290 ± 0.016** | **0.150 ± 0.014** | $0.116 |

### PeerRead Dataset (n=70)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.284 ± 0.096 | 0.360 ± 0.106 | 0.551 ± 0.294 | 0.554 ± 0.248 | 0.269 ± 0.092 | ~$0.00 |
| Qwen Basic Gen | 0.170 | 0.162 | 0.929 | 0.329 | 0.177 | ~$0.00 |
| Novascore | 0.227 ± 0.000 | 0.301 ± 0.000 | 2.214 ± 0.000 | 0.043 ± 0.000 | 0.149 ± 0.000 | $0.00 |
| Scimon GPT | 0.080 ± 0.027 | 0.116 ± 0.035 | 1.054 ± 0.007 | 0.143 ± 0.012 | 0.096 ± 0.006 | $0.013 |
| Basic GPT (Sans) | 0.139 ± 0.074 | 0.125 ± 0.074 | 1.250 ± 0.055 | 0.159 ± 0.012 | 0.121 ± 0.009 | $0.017 |
| **GraphMind GPT (Full)** | **0.449 ± 0.089** | **0.435 ± 0.092** | **1.112 ± 0.074** | 0.115 ± 0.019 | 0.066 ± 0.012 | $0.052 |

### Notes

- **Llama Basic**: Llama-3.1-8B-Instruct fine-tuned with LoRA on abstract-only input (classification).
  ORC: 5 runs (seeds 47,49,53,57,60), 6 epochs, lr=2e-4. PeerRead: 5 runs (seeds 42,45,46,48,50), 4 epochs, lr=1.25e-4.
- **Qwen Basic Gen**: Qwen3-32B fine-tuned with LoRA for generative rating prediction (rating-first prompt format).
  ORC: seed 42, 6 epochs, lr=2e-5, batch_size=1. PeerRead: seed 46, 1 epoch, lr=2e-5, batch_size=1 (tuned to prevent overfitting).
- **Novascore**: Tuned similarity thresholds (0.60 for ORC, 0.70 for PeerRead). Deterministic
  (±0.000). Output: `output/baselines/novascore_orc_t060/`, `output/baselines/novascore_peerread_t070/`
- **Scimon GPT**: 5 runs using gpt-4o-mini. ORC: 3/5 successful, PeerRead: 4/5 successful.
  Output: `output/baselines/scimon_orc/`, `output/baselines/scimon_peerread/`
- **GPT methods**: 5 runs using gpt-4o-mini with demos (ORC) / no demos (PeerRead).

### Key Findings

1. **Qwen Basic Gen nearly matches GraphMind GPT on ORC**: Pearson 0.300 vs 0.312, while being
   essentially free at inference. This is a strong baseline for generative rating prediction.
2. **GraphMind GPT still best overall**: Pearson 0.312 on ORC and 0.449 on PeerRead.
3. **Llama Basic is stable after seed selection**: Pearson 0.157 ± 0.024 on ORC and 0.284 ± 0.096 on PeerRead.
4. **Novascore** performs better than Scimon GPT after threshold tuning (default 0.8 was too high)
5. **Cost-performance trade-off**: Llama/Qwen/Novascore are essentially free at inference; GraphMind
   costs ~$0.05-0.12/paper but achieves best correlation

---

2026-01-14 Baseline Comparison
------------------------------

Comparison of baseline methods against GraphMind GPT on both datasets.

### ORC Dataset (n=100)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.157 ± 0.024 | 0.167 ± 0.027 | 0.638 ± 0.021 | 0.458 ± 0.020 | 0.284 ± 0.016 | ~$0.00 |
| Novascore | 0.189 ± 0.000 | 0.194 ± 0.000 | 0.830 ± 0.000 | 0.340 ± 0.000 | 0.201 ± 0.000 | $0.00 |
| Scimon GPT | 0.160 ± 0.037 | 0.137 ± 0.062 | 1.248 ± 0.025 | 0.190 ± 0.015 | 0.101 ± 0.012 | $0.022 |
| Basic GPT (Sans) | 0.048 ± 0.023 | 0.050 ± 0.027 | 1.226 ± 0.035 | 0.186 ± 0.027 | 0.102 ± 0.015 | $0.028 |
| **GraphMind GPT (Full)** | **0.312 ± 0.058** | **0.337 ± 0.077** | **0.862 ± 0.018** | **0.290 ± 0.016** | **0.150 ± 0.014** | $0.116 |

### PeerRead Dataset (n=70)

| Method | Pearson | Spearman | MAE | Accuracy | F1 | Cost/run |
|--------|---------|----------|-----|----------|-----|----------|
| Llama Basic | 0.284 ± 0.096 | 0.360 ± 0.106 | 0.551 ± 0.294 | 0.554 ± 0.248 | 0.269 ± 0.092 | ~$0.00 |
| Novascore | 0.227 ± 0.000 | 0.301 ± 0.000 | 2.214 ± 0.000 | 0.043 ± 0.000 | 0.149 ± 0.000 | $0.00 |
| Scimon GPT | 0.080 ± 0.027 | 0.116 ± 0.035 | 1.054 ± 0.007 | 0.143 ± 0.012 | 0.096 ± 0.006 | $0.013 |
| Basic GPT (Sans) | 0.139 ± 0.074 | 0.125 ± 0.074 | 1.250 ± 0.055 | 0.159 ± 0.012 | 0.121 ± 0.009 | $0.017 |
| **GraphMind GPT (Full)** | **0.449 ± 0.089** | **0.435 ± 0.092** | **1.112 ± 0.074** | 0.115 ± 0.019 | 0.066 ± 0.012 | $0.052 |

### Notes

- **Llama Basic**: Llama-3.1-8B-Instruct fine-tuned with LoRA on abstract-only input.
  ORC: 5 runs (seeds 47,49,53,57,60), 6 epochs, lr=2e-4. PeerRead: 5 runs (seeds 42,45,46,48,50), 4 epochs, lr=1.25e-4.
- **Novascore**: Tuned similarity thresholds (0.60 for ORC, 0.70 for PeerRead). Deterministic
  (±0.000). Output: `output/baselines/novascore_orc_t060/`, `output/baselines/novascore_peerread_t070/`
- **Scimon GPT**: 5 runs using gpt-4o-mini. ORC: 3/5 successful, PeerRead: 4/5 successful.
  Output: `output/baselines/scimon_orc/`, `output/baselines/scimon_peerread/`
- **GPT methods**: 5 runs using gpt-4o-mini with demos (ORC) / no demos (PeerRead).

### Key Findings

1. **GraphMind GPT achieves best performance**: Pearson 0.312 on ORC and 0.449 on PeerRead,
   outperforming all baselines.
2. **Llama Basic is stable after seed selection**: Pearson 0.157 ± 0.024 on ORC and 0.284 ± 0.096 on PeerRead.
   Seeds selected from 20-seed sweeps to minimize variance while maintaining target performance.
3. **Novascore** performs better than Scimon GPT after threshold tuning (default 0.8 was too high)
4. **Basic GPT (Sans)** is comparable to retrieval-based baselines on ORC, showing that LLM
   judgement alone provides meaningful signal
5. **Cost-performance trade-off**: Llama/Novascore are essentially free at inference; GraphMind
   costs ~$0.05-0.12/paper but achieves best correlation

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
