# Experiment Documentation

## Validating Prompt Changes

After modifying prompt templates in `src/paper/gpt/prompts/`, run test experiments with
`--limit 1` to verify prompts are building correctly before running full experiments:

```bash
uv run paper gpt eval graph run \
  --papers output/venus5/split/dev_100_balanced.json.zst \
  --output /tmp/claude/test_<config_name> \
  --model gpt-4o-mini --limit 1 \
  --eval-prompt <prompt_name> \
  --demos orc_balanced_4 --seed 42 --n-evaluations 1
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

Base command for running ablation experiments (use `experiment` with `--runs 5` for
stable statistics):
```bash
uv run paper gpt eval graph experiment \
  --papers output/venus5/split/dev_100_balanced.json.zst \
  --output output/eval_orc/ablation_<name> \
  --model gpt-4o-mini --limit 100 --runs 5 \
  --eval-prompt <prompt> \
  --demos orc_balanced_4 --seed 42
```

Add `--sources citations` or `--sources semantic` for source-filtered experiments.

**IMPORTANT**: When running multiple experiments in parallel, limit to a maximum of 3
concurrent experiments to avoid API rate limits.

Expected cost: ~$0.90-1.00 per run x 5 runs = ~$4.50-5.00 per experiment.

## Experiment Log Format

Log experiments to `EXPERIMENT_LOG.yaml` using this format:

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

Update `MAJOR_RESULTS.md` when completing a full set of major experiments (ablation or
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

## Remote Jobs with fleche

`fleche` is a utility for running jobs on remote Slurm clusters via SSH. Configuration
is in `fleche.toml`.

### Quick Start

```bash
# Validate configuration
fleche check

# Preview what would be submitted (recommended first)
fleche run <job> --dry-run

# Submit a job (streams output by default)
fleche run <job>

# Submit without streaming (returns immediately)
fleche run <job> --bg
```

### Available Jobs

| Job | Description | Command |
|-----|-------------|---------|
| `cuda_check` | Test CUDA/GPU availability | `fleche run cuda_check` |
| `train` | Train Llama SFT model | `fleche run train --env DATASET=orc` |
| `infer` | Run inference | `fleche run infer --env DATASET=orc` |

### Running Llama Experiments

**Train on ORC:**
```bash
fleche run train --env DATASET=orc --env CONFIG=llama_basic \
  --tag dataset=orc --tag config=llama_basic
```

**Train on PeerRead:**
```bash
fleche run train --env DATASET=peerread --env CONFIG=llama_peerread \
  --tag dataset=peerread --tag config=llama_peerread
```

**Run inference after training:**
```bash
fleche run infer --env DATASET=orc --env CONFIG=llama_basic \
  --tag dataset=orc --tag type=inference
```

Jobs share a workspace, so the trained model from `train` is automatically available
to `infer` without needing to download and re-upload.

### Tagging Jobs

Use tags to organise and filter experiments:

```bash
# Add tags when submitting
fleche run train --env DATASET=orc --tag experiment=ablation --tag model=llama

# Filter status by tag
fleche status --tag experiment=ablation
fleche status --tag dataset=orc --filter running

# View logs from most recent job with specific tag
fleche logs --tag experiment=ablation

# Download outputs from most recent job with tag
fleche download --tag config=llama_basic

# Cancel all jobs with a specific tag
fleche cancel --all --tag experiment=test

# Clean old jobs with a specific tag
fleche clean --all --tag experiment=test
```

### Monitoring and Results

```bash
# Check job status
fleche status

# Show last N jobs
fleche status -n 20

# Filter by status (running, pending, completed, failed, cancelled)
fleche status --filter running

# List all unique tags
fleche tags

# View logs (defaults to most recent job)
fleche logs

# View logs for specific job
fleche logs <job-id>

# Show only the last N lines
fleche logs -n 50

# Show only stdout or stderr
fleche logs --stdout
fleche logs --stderr

# Stream logs in real-time (Ctrl+C to disconnect, job keeps running)
fleche logs --follow

# Download results after completion (defaults to most recent job)
fleche download

# Re-run a previous job with same settings
fleche rerun <job-id>

# Cancel a running job (defaults to most recent active)
fleche cancel

# Cancel all running/pending jobs
fleche cancel --all
```

### Quick Commands Without Slurm

For quick tests or non-GPU work, use `exec` to bypass the Slurm queue:

```bash
fleche exec "ls -la"
fleche exec "python test.py"
```

### File Sync

- Project code syncs automatically to shared workspace (respects `.gitignore`)
- Input data specified in `inputs` is copied to workspace
- Jobs share workspace, so outputs from one job are available to subsequent jobs
- Use `fleche download` to pull outputs to local machine

### Tips

- Use `--dry-run` to preview the sbatch script before submitting
- Ctrl+C during streaming disconnects but doesn't cancel the job
- Job IDs look like `train-20260114-153042-847-x7k2`
- Use `fleche exec` for quick tests without Slurm queue wait
- Run `fleche guide` for full documentation
