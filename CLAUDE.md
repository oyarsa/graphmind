# Codebase Guidelines

## Workflow Rules
- **ALWAYS run `just lint`** after code changes before considering the task complete
- **Run backend e2e tests** (`uv run pytest -m backend --runslow`) after major backend/API changes before declaring done
- If you find yourself manually testing behaviours (curl, `call_sse.py`, etc.), consider adding tests in `tests/e2e/test_backend_evaluation.py`
- **ALWAYS notify the user** when done, even without needing input
- **Log ALL experiments** to `labs/EXPERIMENT_LOG.yaml` including:
  - GPT ablation experiments
  - Llama/SFT baseline experiments (record config, seeds, learning rate, epochs)
  - Any other baseline experiments (Novascore, Scimon, etc.)
- **Record ALL metrics** from `paper.evaluation_metrics.Metrics` in the log:
  - `pearson`, `spearman` - correlation metrics
  - `mae`, `mse`, `rmse` - error metrics
  - `accuracy`, `acc_pm1` (accuracy_within_1) - accuracy metrics
  - `f1`, `precision`, `recall` - classification metrics
  - `cost_per_run` - if applicable
- Update `labs/MAJOR_RESULTS.md` for major results (see `docs/EXPERIMENTS.md` for format)

## Commands
- `just lint` - Main check: fix, fmt, spell, pre-commit, test, type
- `just test` - Unit tests only
- `just e2e` - E2E tests (with --runslow)
- `uv run pytest -m backend --runslow` - Backend API e2e tests only
- `uv run paper [subcommand] --help` - CLI help

## API Server

**Start the server:**
```bash
just api-dev
```
Server runs at http://127.0.0.1:8000 with docs at http://127.0.0.1:8000/docs

**Test SSE endpoints:**
```bash
# Call endpoint and save response (shows progress spinner)
uv run python scripts/call_sse.py "http://127.0.0.1:8000/mind/evaluate?id=2406.18245v2&title=..." /tmp/out.json

# Inspect specific fields
cat /tmp/out.json | jq '.result.result.paper.structured_evaluation | {label, supporting: [.supporting_evidence[].source], contradictory: [.contradictory_evidence[].source]}'
```

**Common test cases:**

1. Evidence distribution (3 semantic + fill with citations):
   ```bash
   uv run python scripts/call_sse.py \
     "http://127.0.0.1:8000/mind/evaluate?id=2406.18245v2&title=Weak%20Reward%20Model&llm_model=gpt-4o-mini" \
     /tmp/evidence.json
   cat /tmp/evidence.json | jq '{
     supporting: [.result.result.paper.structured_evaluation.supporting_evidence[].source],
     contradictory: [.result.result.paper.structured_evaluation.contradictory_evidence[].source]
   }'
   # Expected: up to 3 "semantic" followed by "citations" (max 5 total per array)
   ```

2. Basic evaluation check:
   ```bash
   cat /tmp/out.json | jq '.result.result.paper.structured_evaluation | {label, has_summary: (.paper_summary | length > 0)}'
   ```

**Response structure** (for `/mind/evaluate` and `/mind/evaluate-abstract`):
- `result.result.paper.structured_evaluation.label` - Novelty score (1-5)
- `result.result.paper.structured_evaluation.paper_summary` - Summary of contributions
- `result.result.paper.structured_evaluation.supporting_evidence[]` - Evidence items with `.source`, `.text`, `.paper_title`
- `result.result.paper.structured_evaluation.contradictory_evidence[]` - Same structure
- `result.result.paper.structured_evaluation.key_comparisons[]` - Technical comparisons
- `result.result.paper.structured_evaluation.conclusion` - Final assessment

## Code Style
- **All files must be shorter than 1000 non-empty lines** (blank lines don't count). If a file exceeds this limit, split it into smaller, well-organized modules.
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
- **Check `fleche.toml` first** for available jobs
- Remote Slurm jobs: `train`, `infer`, `train_gen`, `infer_gen`, `train_gen_graph`, `infer_gen_graph`
- Local GPT jobs: `gpt_orc`, `gpt_peerread`, `gpt_orc_test`, `gpt_peerread_test`
- Most commands default to most recent job if no job-id given
- Short ID suffix works (e.g., `x7k2` instead of full `train-20260115-153042-847-x7k2`)
- Config supports `${VAR}` substitution from env vars, `.env` file, and `${PROJECT}` built-in

**Running jobs:**
- `fleche run <job>` - Submit and stream output (Ctrl+C disconnects, job keeps running)
- `fleche run <job> --bg` - Run in background (`--notify` for alerts)
- `fleche run <job> --env VAR=value --tag key=value` - Set env vars and tags
- `fleche run <job> --note "description"` - Add note to document experiment
- `fleche run <job> --command "nvidia-smi"` - Override command (keeps job's Slurm config)
- `fleche run <job> --dry-run` - Preview sbatch script without submitting
- `fleche run <job> --host local` - Run locally instead of on remote Slurm cluster
- `fleche run <job> --after <job-id>` - Run after another job completes (dependency)
- `fleche run <job> --retry 3` - Auto-retry on failure with exponential backoff
- `fleche exec <cmd>` - Run directly via SSH, no Slurm (quick tests)
- `fleche exec <cmd> --host local` - Run command locally without SSH
- `fleche run "command" --gpus 1 --time 1:00:00` - Adhoc Slurm command (no job definition)
- `fleche rerun <job-id>` - Re-run previous job with same settings

**Monitoring:**
- `fleche status -n 20` - Show last 20 jobs
  - `--filter running` - Filter by status (running/pending/completed/failed/cancelled)
  - `--tag key=value` - Filter by tag
  - `--name 'pattern'` - Filter by job ID regex (substring match, use `^`/`$` to anchor)
  - `--archived` - Show only archived jobs
  - `--all-jobs` - Show all jobs including archived
- `fleche logs [job-id]` - View logs (`--raw` to strip ANSI, `--follow` to stream)
  - `-n 50` - Show only last N lines
  - `--stdout` / `--stderr` - Show only one stream
  - `--note 'pattern'` - Filter by note content (case-insensitive regex)
- `fleche wait [job-id]` - Wait for completion (`--notify` for alerts)
- `fleche stats [job-id]` - Show resource usage (elapsed time, CPU time, max memory)
- `fleche note <job-id> [text]` - View or set job note
- `fleche ping` - Check Slurm cluster health
- `fleche check` - Validate config after editing
- `fleche check --remote` - Validate config against remote server (SSH, Slurm, disk space)
- `fleche doctor` - Comprehensive troubleshooting diagnostics
- `fleche compare <a> <b>` - Compare two job configurations side-by-side

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
- `fleche clean --archive <job-id>` - Archive job (hide without deleting)
- `fleche clean --unarchive <job-id>` - Restore archived job

**GPT experiments (local):**
- `fleche run gpt_orc --env PROMPT=sans` - Run ORC ablation with specific prompt
- `fleche run gpt_peerread --env PROMPT=full-graph-structured` - Run PeerRead ablation
- `fleche run gpt_orc_test --env PROMPT=sans` - Quick test (1 paper, 1 run)
- Prompts: `sans`, `related`, `norel-graph`, `semantic-only`, `full-graph-structured`
- Add `--env SOURCES=citations` or `--env SOURCES=semantic` for source filtering
- Add `--env RUNS=1` for single run, `--env LIMIT=10` to limit papers

## Environment
Create `.env` from `.env.example` with `OPENAI_API_KEY` and optionally `SEMANTIC_SCHOLAR_API_KEY`.

## Data Formats
- JSON with optional compression (`.json.gz`, `.json.zst`)
- Query compressed files: `zstd -dc file.json.zst | jq 'filter'`
- Prompts: `src/paper/gpt/prompts/`
- Demos: `src/paper/gpt/demonstrations/`

## Experiment Setup

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
