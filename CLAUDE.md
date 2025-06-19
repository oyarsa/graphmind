# CLAUDE.md - Codebase Guidelines

## Important Workflow Rules
**ALWAYS run `just lint` after making any code changes before considering the task
complete.** This ensures code formatting, type checking, and all tests pass.

**ALWAYS notify the user when you are done, even if you don't need input from them.**

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
- **Type Access**: Never use hasattr/getattr. Always get the proper type and access fields through there.

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
- **TOML**: Configuration files and prompts (`src/paper/gpt/prompts/*.toml`)
- **Demonstrations**: JSON format in `src/paper/gpt/demonstrations/`
- **Config**: Baseline configurations in `src/paper/baselines/sft_config/`

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
