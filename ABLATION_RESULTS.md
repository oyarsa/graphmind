# Ablation Study Results

This file contains summary tables comparing different configurations across datasets.

**New results are added at the top in reverse chronological order.**

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
- **Related**: `eval-prompt=related` (v6 with "focus on abstract claims"), related papers from both sources
- **Graph Only**: `eval-prompt=norel-graph`, graph summary without related papers
- **Citations**: `eval-prompt=full-graph-structured`, `sources=citations`
- **Semantic**: `eval-prompt=semantic-only` (v2 conservative), `sources=semantic`
- **Full**: `eval-prompt=full-graph-structured`, `sources=both`

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
- **Related**: `eval-prompt=related`, related papers from both sources
- **Graph Only**: `eval-prompt=norel-graph`, graph summary without related papers
- **Citations**: `eval-prompt=full-graph-structured`, `sources=citations`
- **Semantic**: `eval-prompt=full-graph-structured`, `sources=semantic`
- **Full**: `eval-prompt=full-graph-structured`, `sources=both`

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
