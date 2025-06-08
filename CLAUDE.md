# CLAUDE.md - Codebase Guidelines

## Important Workflow Rules
**ALWAYS run `just lint` after making any code changes before considering the task
complete.** This ensures code formatting, type checking, and all tests pass.

**ALWAYS notify the user when you are done, even if you don't need input from them.**

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

The main command to use to check if the code is correct is `just lint`.

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

## Version control
- Use jujutsu (jj) instead of git to manage version control in this repository.
- After each major operation, create a new revision.
- In general, follow all the common git workflows, but using jujutsu instead. If you
  don't know how to do something, ask and I'll help you.
