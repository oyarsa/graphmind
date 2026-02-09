# Experiment Documentation

## Initial Setup

When initialising the repo for experiments (with `cpu` or `cuda`):

1. **Install dependencies** (cpu and cuda extras are mutually exclusive):
   ```bash
   uv sync --extra baselines --extra cuda   # For GPU hosts
   uv sync --extra baselines --extra cpu    # For CPU-only hosts
   ```

2. **Check for experiment data** - verify these paths exist:
   - `output/venus5/split/dev_100_balanced.json.zst` (ORC test data)
   - `output/new_peerread/peter_summarised/balanced_68.json.zst` (PeerRead test data)
   - `output/baselines/llama_data/` (Llama train/dev/test)
   - `output/baselines/orc_acu_query_t05/` (Novascore ORC queries)
   - `output/baselines/peerread_acu_query_t05/` (Novascore PeerRead queries)

3. **If data is missing**, tell the user:
   > The experiment data (~44 MB tarball) is not included in the repo.
   > Please obtain `experiment_data.tar.gz` and extract it in the repo root:
   > ```bash
   > tar -xzf experiment_data.tar.gz
   > ```
   > To create this tarball from a host that has the data: `bash tmp/create_data_tarball.sh`

4. **Create `.env`** from `.env.example` with `OPENAI_API_KEY` (required for GPT experiments)

5. **Verify setup**:
   ```bash
   just lint                    # Should pass
   uv run paper --help          # Should show CLI help
   ```

6. **For CUDA hosts**, verify GPU access:
   ```bash
   uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

## Validating Prompt Changes

After modifying prompt templates in `src/paper/gpt/prompts/`, run test experiments with
`--limit 1` to verify prompts are building correctly before running full experiments:

```bash
# Quick test via fleche (1 paper, 1 run)
fleche run gpt_orc_test --env PROMPT=<prompt_name>
fleche run gpt_peerread_test --env PROMPT=<prompt_name>
```

For custom options not covered by fleche, use the underlying command:
```bash
uv run paper gpt eval graph run --help
```

Common prompts to test:
- `sans` - Abstract only (baseline, ~1300 tokens)
- `related` - Related papers without graph (~3800 tokens)
- `norel-graph` - Graph without related papers
- `semantic-only` - Graph + semantic papers only (conservative about semantic matches)
- `full-graph-structured` - Full pipeline (~30000 tokens)

For ablation experiments with source filtering, add `--sources citations` or
`--sources semantic`.

## Ablation Experiments

Standard ablation study configurations for measuring component contributions on the
ORC 100-item balanced dataset:

| Name           | Eval Prompt             | Sources   | Description                    |
|----------------|-------------------------|-----------|--------------------------------|
| Sans           | `sans`                  | N/A       | Abstract only (baseline)       |
| Related only   | `related`               | both      | Related papers without graph   |
| Graph only     | `norel-graph`           | N/A       | Graph without related papers   |
| Citations only | `full-graph-structured` | citations | Full pipeline, citations only  |
| Semantic only  | `semantic-only`         | semantic  | Graph + semantic (conservative)|
| Full           | `full-graph-structured` | both      | Full pipeline (baseline)       |

Run ablation experiments via fleche (5 runs by default):
```bash
# ORC dataset (100 papers)
fleche run gpt_orc --env PROMPT=sans
fleche run gpt_orc --env PROMPT=full-graph-structured --env SOURCES=citations

# PeerRead dataset (68 papers)
fleche run gpt_peerread --env PROMPT=sans
fleche run gpt_peerread --env PROMPT=semantic-only --env SOURCES=semantic
```

For custom options not covered by fleche, use the underlying command:
```bash
uv run paper gpt eval graph experiment --help
```

**IMPORTANT**: When running multiple experiments in parallel, limit to a maximum of 3
concurrent experiments to avoid API rate limits.

Expected cost: ~$0.90-1.00 per run x 5 runs = ~$4.50-5.00 per experiment.

## Experiment Log Format

Log experiments to `labs/EXPERIMENT_LOG.yaml` using this format:

```yaml
- date: "YYYY-MM-DD"
  name: "experiment_name"
  description: "Short description"
  reason: "Why you ran this experiment (1-2 sentences)"
  command: |
    Full command used
  parameters:
    dataset: "dataset name"
    model: "model name"
    eval_prompt: "prompt name"
    sources: "citations/semantic/null"
    demos: "demo set"
    runs: N
  metrics:
    pearson: {mean: X, stdev: X, min: X, max: X}
    spearman: {mean: X, stdev: X, min: X, max: X}
    mae: {mean: X, stdev: X, min: X, max: X}
    accuracy: {mean: X, stdev: X, min: X, max: X}
    f1: {mean: X, stdev: X, min: X, max: X}
    cost_per_run: X
  total_cost: X
  conclusion: "What you learned (1-2 sentences)"
```

New experiments are appended at the bottom.

## Ablation Results Summary (MAJOR_RESULTS.md)

Update `labs/MAJOR_RESULTS.md` when completing a full set of major experiments (ablation or
baseline comparisons). Format:

1. **Add new results at the TOP** (reverse chronological order)
2. **Timestamp heading**: Use H2 with underline syntax (`---`)
   ```markdown
   YYYY-MM-DD HH:MM (ORC) / HH:MM (PeerRead)
   ------------------------------------------
   ```
3. **Include tables for each dataset**: ORC and PeerRead (use H3 `###`)
4. **Configuration Details section**: List each configuration with its settings AND output
   directory path (e.g., `Output: output/eval_orc/ablation_sans/`)
5. **Common Settings section**: Include model config, dataset details with **file hashes**,
   and key findings
6. **Keep old results**: Never delete previous entries

To get file hash from experiment output:
```bash
cat output/eval_<dataset>/ablation_<config>/run_0/params.json | grep paper_file
# Extract hash from: "file.json.zst (12345678)"
```

## Remote Jobs with Fleche

See `docs/FLECHE.md` for the full fleche reference (all commands, flags, and examples).
