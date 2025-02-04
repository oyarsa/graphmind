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

# Run pyright
type:
    uv run pyright .

# Run pre-commit hooks manually on all files
pre-commit:
    uv run pre-commit run --all-files

# Run codespell on code file to find typos
spell:
    uv run codespell

# Run ruff check, ruff format, and pyright
lint: fix fmt spell pre-commit test type

# Check ruff lint and pyright
check-all: check test spell type

# Watch Python files and run ruff and pyright on changes
watch:
    watchexec --exts=py --clear --restart "just check-all"

# Generate CLI documentation files
clidoc:
    #!/usr/bin/env bash
    for module in gpt peter scimon peerread; do
        uv run typer "paper.$module.cli" utils docs --output "src/paper/$module/CLI.md" &
    done
    wait

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
    uv run pyright . | grep -o '/.*\.py' | sort | uniq -c | sort -n

alias l := lint
alias w := watch
alias x := exp
