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
- `fleche run <job> --bg` - Run job in background (use `--notify` for alerts)
- `fleche run <job> --env VAR=value` - Set environment variables
- `fleche run <job> --tag key=value` - Tag jobs for filtering
- `fleche status -n 20` - Show last 20 jobs (`--tag k=v` by tags, `--name '*pattern*'` by job name glob)
- `fleche logs <job-id>` - View job logs (supports short ID suffix matching)
- `fleche wait <job-id>` - Wait for job to finish (add `--notify` for alerts)
- `fleche rerun <job-id>` - Re-run a previous job with same settings
- `fleche ping` - Check cluster health
- Jobs defined in `fleche.toml`: `train`, `infer`, `train_gen`, `infer_gen`

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
