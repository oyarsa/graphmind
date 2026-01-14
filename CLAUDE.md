# CLAUDE.md - Codebase Guidelines

## Workflow Rules
- **ALWAYS run `just lint`** after code changes before considering the task complete
- **ALWAYS notify the user** when done, even without needing input
- **Log experiments** to `EXPERIMENT_LOG.yaml` and update `MAJOR_RESULTS.md` for major results (see `docs/EXPERIMENTS.md` for format)

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

## Environment
Create `.env` from `.env.example` with `OPENAI_API_KEY` and optionally `SEMANTIC_SCHOLAR_API_KEY`.

## Data Formats
- JSON with optional compression (`.json.gz`, `.json.zst`)
- Query compressed files: `zstd -dc file.json.zst | jq 'filter'`
- Prompts: `src/paper/gpt/prompts/`
- Demos: `src/paper/gpt/demonstrations/`
