# CLAUDE.md - Codebase Guidelines

## Workflow Rules
- **ALWAYS run `just lint`** after code changes before considering the task complete
- **ALWAYS notify the user** when done, even without needing input
- **Log ALL experiments** to `EXPERIMENT_LOG.yaml` including:
  - GPT ablation experiments
  - Llama/SFT baseline experiments (record config, seeds, learning rate, epochs)
  - Any other baseline experiments (Novascore, Scimon, etc.)
- Update `MAJOR_RESULTS.md` for major results (see `docs/EXPERIMENTS.md` for format)

## Commands
- `just lint` - Main check: fix, fmt, spell, pre-commit, test, type
- `just test` - Unit tests only
- `just e2e` - E2E tests (with --runslow)
- `uv run paper [subcommand] --help` - CLI help

## Code Style
- Python 3.12+ with strict typing (no exceptions)
- Functional programming with dataclasses over OOP
- British English everywhere
- Absolute imports from `paper` package
- Google-style docstrings
- Never use hasattr/getattr - get proper type and access fields

## Version Control
- Use **jujutsu** (`jj`), not git
- `jj commit -m "<message>"` to commit (no staging needed)
- Only commit when requested
- First line of commit messages <= 69 chars

## Reference Documentation
- **Experiments**: `docs/EXPERIMENTS.md` - ablation configs, experiment log format, prompt validation
- **Baselines**: `docs/BASELINES.md` - SFT/Llama fine-tuning setup

## Fleche (Remote Job Submission)
- **Check `fleche.toml` first** for available jobs (`train`, `infer`, `train_gen`, `infer_gen`)
- Most commands default to most recent job if no job-id given
- Short ID suffix works (e.g., `x7k2` instead of full `train-20260115-153042-847-x7k2`)
- Config supports `${VAR}` substitution from env vars, `.env` file, and `${PROJECT}` built-in

**Running jobs:**
- `fleche run <job>` - Submit and stream output (Ctrl+C disconnects, job keeps running)
- `fleche run <job> --bg` - Run in background (`--notify` for alerts)
- `fleche run <job> --env VAR=value --tag key=value` - Set env vars and tags
- `fleche run <job> --command "nvidia-smi"` - Override command (keeps job's Slurm config)
- `fleche run <job> --dry-run` - Preview sbatch script without submitting
- `fleche run <job> --host local` - Run locally instead of on remote Slurm cluster
- `fleche exec <cmd>` - Run directly via SSH, no Slurm (quick tests)
- `fleche exec <cmd> --host local` - Run command locally without SSH
- `fleche run "command" --gpus 1 --time 1:00:00` - Adhoc Slurm command (no job definition)
- `fleche rerun <job-id>` - Re-run previous job with same settings

**Monitoring:**
- `fleche status -n 20` - Show last 20 jobs
  - `--filter running` - Filter by status (running/pending/completed/failed/cancelled)
  - `--tag key=value` - Filter by tag
  - `--name 'pattern'` - Filter by job ID regex (substring match, use `^`/`$` to anchor)
- `fleche logs [job-id]` - View logs (`--raw` to strip ANSI, `--follow` to stream)
  - `-n 50` - Show only last N lines
  - `--stdout` / `--stderr` - Show only one stream
- `fleche wait [job-id]` - Wait for completion (`--notify` for alerts)
- `fleche ping` - Check Slurm cluster health
- `fleche check` - Validate config after editing

**Results:**
- `fleche download [job-id]` - Download output files (`--partial` while job running)
  - `--filter "*.json"` - Download only specific file types (repeatable, searches recursively inside directories)
  - `--filter "!checkpoints/**"` - Exclude files/directories with `!` prefix
  - `--dry-run` - Preview what would be downloaded without actually downloading
  - Example: `fleche download --filter "*.json" --filter "*.json.zst" --filter "*.log" --filter "!checkpoint*/**"` - Download outputs only, skip model weights
- `fleche tags` - List unique tags across all jobs

**Cleanup:**
- `fleche cancel [job-id]` - Cancel job (`--all` for all active, `--tag` to filter)
- `fleche clean --older-than 2h -y` - Clean old jobs periodically
- `fleche clean --workspace` - Also delete shared workspace (use with caution)

## Environment
Create `.env` from `.env.example` with `OPENAI_API_KEY` and optionally `SEMANTIC_SCHOLAR_API_KEY`.

## Data Formats
- JSON with optional compression (`.json.gz`, `.json.zst`)
- Query compressed files: `zstd -dc file.json.zst | jq 'filter'`
- Prompts: `src/paper/gpt/prompts/`
- Demos: `src/paper/gpt/demonstrations/`

## Experiment Setup (for Claude)

When the user asks to "initialise the repo for experiments" (with `cpu` or `cuda`), follow these steps:

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
