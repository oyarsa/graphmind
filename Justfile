_default:
    @just --list

# Run ruff check with autofix
fix:
    uv run ruff check . --fix

# Run ruff check (no fix)
check:
    uv run ruff check .

# Run ruff format
fmt:
    uv run ruff format .

# Run basedpyright (type checker)
type:
    uv run basedpyright .

# Run pre-commit hooks manually on all files
pre-commit:
    uv run pre-commit run --all-files

# Run codespell on code file to find typos
spell:
    uv run codespell

# Run ruff check, format, spell checker, pre-commit, tests and type checker
lint: fix fmt spell pre-commit test type

# Check ruff check, tests, spell checker and type checker
check-all: check test spell type

# Watch Python files and run `check-all` on changes
watch:
    watchexec --exts=py --clear --restart "just check-all"

# Run end-to-end tests with pytest
e2e:
    uv run pytest -s --runslow tests/

# Run unit tests only
test:
    uv run pytest --quiet tests/

# Run experiments (see experiments/Justfile)
exp *args:
    @just -f experiments/Justfile {{args}}

# Show all files with type errors
typefiles:
    uv run basedpyright . | grep -o '/.*\.py' | sort | uniq -c | sort -n

alias l := lint
alias w := watch
alias x := exp
