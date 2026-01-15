# SFT/Llama Baseline Experiments

Fine-tune Llama 3.1-8B-Instruct using LoRA for originality rating prediction. Requires
GPU with CUDA support (will not work on macOS).

## Prerequisites

1. Install baselines dependencies: `uv sync --extra baselines --extra cuda`
2. Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. HuggingFace access to Llama model (may require `huggingface-cli login`)

## Data Files

Pre-prepared data in `output/baselines/llama_data/`:

**ORC (basic mode - title + abstract):**
| File | Samples | Notes |
|------|---------|-------|
| `orc_train.json.zst` | 200 | Balanced by rating |
| `orc_dev.json.zst` | 100 | Balanced by rating |
| `orc_test.json.zst` | 100 | Same as GPT ablation test set |

**PeerRead (basic mode - title + abstract):**
| File | Samples | Notes |
|------|---------|-------|
| `peerread_train.json.zst` | 70 | From remaining data |
| `peerread_dev.json.zst` | 30 | From remaining data |
| `peerread_test.json.zst` | 70 | Same as GPT ablation test set |

**Graph mode (test only):**
- `orc_test_graph.json.zst` - 100 samples with graph + related papers
- `peerread_test_graph.json.zst` - 68 samples with graph + related papers

Note: Graph-mode train/dev data requires running the expensive GPT graph extraction
pipeline. Currently only test sets have graph data.

## Running Experiments

**ORC Basic:**
```bash
uv run paper baselines sft train \
  --train output/baselines/llama_data/orc_train.json.zst \
  --dev output/baselines/llama_data/orc_dev.json.zst \
  --test output/baselines/llama_data/orc_test.json.zst \
  --output output/baselines/llama_orc_basic \
  --config src/paper/baselines/sft_config/llama_basic.toml
```

**PeerRead Basic:**
```bash
uv run paper baselines sft train \
  --train output/baselines/llama_data/peerread_train.json.zst \
  --dev output/baselines/llama_data/peerread_dev.json.zst \
  --test output/baselines/llama_data/peerread_test.json.zst \
  --output output/baselines/llama_peerread_basic \
  --config src/paper/baselines/sft_config/llama_basic.toml
```

## Output Structure

Each experiment creates:
- `final_model/` - Trained LoRA adapter weights
- `evaluation_metrics.txt` - Accuracy, MAE, MSE metrics
- `predictions.json.zst` - Per-sample predictions
- `config.toml` - Copy of configuration used

## Configuration

Config files in `src/paper/baselines/sft_config/`:
- `llama_basic.toml` - Basic mode (max_length=512)
- `llama_graph.toml` - Graph mode (max_length=2000)

Key settings: LoRA r=8, alpha=16, batch_size=16, lr=1e-4, epochs=1, 4-bit quantisation.

## Remote Execution with fleche

For running on a Slurm cluster, use `fleche` (see `fleche.toml` for configuration):

```bash
# Train on ORC
fleche run train --env DATASET=orc --env CONFIG=llama_basic

# Train on PeerRead
fleche run train --env DATASET=peerread --env CONFIG=llama_basic

# Run inference after training
fleche run infer --env DATASET=orc --env CONFIG=llama_basic
fleche run infer --env DATASET=peerread --env CONFIG=llama_basic

# Monitor and download results
fleche status
fleche logs <job-id> --follow
fleche sync <job-id>
```

See `docs/EXPERIMENTS.md` for more fleche commands.
