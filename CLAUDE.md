# CLAUDE.md - Codebase Guidelines

## Important Workflow Rules
**ALWAYS run `just lint` after making any code changes before considering the task
complete.** This ensures code formatting, type checking, and all tests pass.

**ALWAYS notify the user when you are done, even if you don't need input from them. This includes when exiting plan mode.**

## Environment Variables
Create a `.env` file from `.env.example` with the following:
- `OPENAI_API_KEY` - Required for GPT operations (get from https://platform.openai.com/api-keys)
- `OPENAI_BASE_URL` - Optional alternative API endpoint
- `SEMANTIC_SCHOLAR_API_KEY` - For Semantic Scholar API access (get from https://www.semanticscholar.org/product/api#api-key-form)
- `LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL), default: INFO

## Build & Test Commands
- `just fmt` - Format code with ruff
- `just check` - Run ruff check without fixes
- `just fix` - Run ruff check with autofix
- `just type` - Run pyright type checking
- `just lint` - Run fix, fmt, spell, pre-commit, test, type
- `just test` - Run unit tests only
- `just e2e` - Run end-to-end tests (with --runslow flag)
- `just watch` - Watch for file changes and run checks
- `uv run pytest tests/path/to/test_file.py::test_function` - Run specific test
- `uv run python path/to/script.py` - Run Python scripts (no need to make them executable)

## Testing
- **Unit tests**: `just test` or `uv run pytest tests/`
- **E2E tests**: `just e2e` or `uv run pytest --runslow tests/`
- **Custom markers**: `@pytest.mark.slow` for long-running tests
- **Test configuration**: Extended settings in `tests/pyproject.toml`
- **Running specific tests**: `uv run pytest tests/path/to/test.py::test_function`

The main command to use to check if the code is correct is `just lint`.

## Validating Prompt Changes
After modifying prompt templates in `src/paper/gpt/prompts/`, run test experiments with
`--limit 1` to verify prompts are building correctly before running full experiments:

```bash
# Test different prompt configurations
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

Expected cost: ~$0.90-1.00 per run × 5 runs = ~$4.50-5.00 per experiment.

## Experiment Log
**ALWAYS log experiments to `EXPERIMENT_LOG.yaml`** when running ablation or prompt
engineering experiments. This YAML format enables programmatic analysis of experiment
history. Each entry should include:

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

New experiments are appended at the bottom. This allows future sessions to quickly
review and analyse recent experiments programmatically.

## Ablation Results Summary
**ALWAYS update `ABLATION_RESULTS.md`** when completing a full set of ablation experiments
(sans, related, graph-only, citations, semantic, full). This file maintains a history of
ablation results across time. Format:

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
6. **Keep old results**: Never delete previous entries, they provide historical comparison

To get file hash from experiment output:
```bash
cat output/eval_<dataset>/ablation_<config>/run_0/params.json | grep paper_file
# Extract hash from: "file.json.zst (12345678)"
```

## External Dependencies
- **pandoc** - Required for LaTeX parsing in ORC dataset processing (install from https://pandoc.org/installing.html)
- **PyTorch** - ML framework used by baseline models (installed via uv, CUDA support on non-macOS systems)
- **fastapi** - Used for the REST API for paper evaluation.

## Code Style
- **Python**: 3.12+ features with strict static typing (no exceptions)
- **Typing**: Type annotations required for all functions/parameters/returns
- **Paradigm**: Focus on functional programming with standalone functions and
  dataclasses instead of object oriented design
- **Naming**: PEP8 conventions (snake_case for variables/functions)
- **Docstrings**: Google style (enforced by Ruff)
- **Imports**: Organized by stdlib, third-party, first-party. Project imports always use
  absolute path.
- **Error handling**: Specific exceptions with context-rich messages
- **Comments**: Avoid adding obvious comments that only describes what the code is
  doing. Make the code as self-documenting as possible. For example, if you have a block
  of code that creates a table, do not add a comment saying "creating table".
- **English**: Always write in British English. This applies to names in the code (e.g.
  classes, functions, etc.) and comments.
- **Type Access**:
  - Never use hasattr/getattr. Always get the proper type and access fields through there.
  - If you really have no other choice, ask for permission first.
- **Testing**: Do not write trivial type checks (e.g. isinstance) when type annotations
  already assert the type. Avoid hasattr/getattr in tests - existence of
  methods/attributes is checked by the type checker.

## CLI Structure
Main command: `uv run paper [subcommand]`
- `peerread` - Process PeerRead dataset (download, preprocess)
- `gpt` - GPT operations (graph extraction, evaluation, annotations)
- `baselines` - Run baseline models (NovaScore, SciMon, PETER)
- `peter` - PETER graph construction operations
- `orc` - ORC dataset operations
- `s2` - Semantic Scholar operations (search, recommendations)
- `split` - Dataset splitting utilities

Use `uv run paper [subcommand] --help` for detailed options.

## Data Formats
- **JSON files**: Primary data format, with optional compression (`.json.gz`, `.json.zst`)
- **JSONL**: Used for streaming large datasets
- **TOML**: Configuration files
- **Python modules**: Prompt templates (`src/paper/gpt/prompts/*.py`)
- **Demonstrations**: JSON format in `src/paper/gpt/demonstrations/`
- **Config**: Baseline configurations in `src/paper/baselines/sft_config/`

### Working with Compressed JSON Files
To query zstd-compressed JSON files, use:
```bash
zstd -dc file.json.zst | jq 'filter'
```

Example - extracting rationales from experiment results:
```bash
# Get first rationale
zstd -dc output/eval_orc/ablation_full/run_0/result.json.zst | jq '.[0].item.paper.rationale_pred'

# Get multiple rationales with paper info
zstd -dc output/eval_orc/ablation_full/run_0/result.json.zst | jq -r '.[0:3] | .[] | "Paper: \(.item.paper.title)\nTrue Score: \(.item.paper.originality_rating)\nPredicted: \(.item.paper.y_pred)\n\nRationale:\n\(.item.paper.rationale_pred)\n\n========\n"'
```

## Project-Specific Patterns
- **Imports**: Always use absolute imports from `paper` package
- **Type-only imports**: Use `if TYPE_CHECKING:` for imports only needed for typing
- **Logging**: Use Python logging instead of print statements
- **Error handling**: Use custom exceptions (e.g., `SerdeError`) with context
- **Data processing**: Support compressed formats automatically (`.json.gz`, `.json.zst`)
- **API keys**: Store in `.env` file (never commit to repository)

## Version control
- Use jujutsu (jj) instead of git to manage version control in this repository.
- After each major operation, create a new revision.
- In general, follow all the common git workflows, but using jujutsu instead. If you
  don't know how to do something, ask and I'll help you.
- Use `jj commit -m "<message>"` to create commits using jujutsu. It works similarly to
  how Git commits work, but you don't need to stage the files first.
- Only commit changes when requested. By default, do not commit anything.
- Ensure that the first line in  commit messages is never more than 69 characters long
- If the message spans multiple lines, make sure there's an empty line between the first
  and the rest.
