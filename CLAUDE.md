# Codebase Guidelines

## Workflow Rules
- **ALWAYS run `just lint`** after code changes before considering the task complete
- **Run backend e2e tests** (`uv run pytest -m backend --runslow`) after major backend/API changes before declaring done
- If you find yourself manually testing behaviours (curl, `call_sse.py`, etc.), consider adding tests in `tests/e2e/test_backend_evaluation.py`
- **ALWAYS notify the user** when done, even without needing input
- **Log ALL experiments** to `labs/EXPERIMENT_LOG.yaml` (see `docs/EXPERIMENTS.md` for format)
- **Record ALL metrics** from `paper.evaluation_metrics.Metrics` in the log:
  - `pearson`, `spearman`, `mae`, `mse`, `rmse`, `accuracy`, `acc_pm1`, `f1`, `precision`, `recall`, `cost_per_run`
- Update `labs/MAJOR_RESULTS.md` for major results (see `docs/EXPERIMENTS.md` for format)

## Commands
- `just lint` - Main check: fix, fmt, spell, pre-commit, test, type
- `just test` - Unit tests only
- `just e2e` - E2E tests (with --runslow)
- `uv run pytest -m backend --runslow` - Backend API e2e tests only
- `uv run paper [subcommand] --help` - CLI help
- `just api-dev` - Start API server at http://127.0.0.1:8000

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

## Environment
- Create `.env` from `.env.example` with `OPENAI_API_KEY` and optionally `SEMANTIC_SCHOLAR_API_KEY`
- JSON with optional compression (`.json.gz`, `.json.zst`); query with `zstd -dc file.json.zst | jq 'filter'`
- Prompts: `src/paper/gpt/prompts/` | Demos: `src/paper/gpt/demonstrations/`

## Reference Documentation
- **API Server**: `docs/API.md` - endpoints, test cases, response structure
- **Experiments**: `docs/EXPERIMENTS.md` - setup, ablation configs, log format, fleche jobs
- **Baselines**: `docs/BASELINES.md` - SFT/Llama fine-tuning setup
- **Fleche**: `docs/FLECHE.md` - remote job submission, all commands and flags
