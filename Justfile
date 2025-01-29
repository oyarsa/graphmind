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

# Run ruff check, ruff format, and pyright
lint: fix fmt type test

# Check ruff lint and pyright
check-all: check type test

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

alias l := lint
alias w := watch
alias x := exp
