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

# Run pyright (type checker)
type:
    uv run pyright .

# Run pre-commit hooks manually on all files
pre-commit:
    test -d .git && uv run pre-commit run --all-files

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

# Run unit tests coverage
cov:
    uv run pytest --cov=paper --cov-report=html --quiet tests/

# Run experiments (see experiments/Justfile)
exp *args:
    @just -f experiments/Justfile {{args}}

# Show all files with type errors
typefiles:
    uv run pyright . | grep -o '/.*\.py' | sort | uniq -c | sort -n

# Show documentation as HTML in a browser
doc:
    uv run pdoc paper --docformat google

# Start development server
api-dev:
    LOG_LEVEL=debug TIMERS=1 uv run fastapi dev src/paper/backend/api.py

# Start production server
api-serve:
    uv run fastapi run src/paper/backend/api.py --port 8001

# Lint both core and frontend
lint-all:
    just lint
    cd frontend && just lint

# Bump version for both backend and frontend (ensures they're synchronized)
version bump="major":
    #!/usr/bin/env bash
    set -euo pipefail

    # Bump the Python version first
    uv version --bump {{bump}}
    uv lock

    # Get the new version from uv
    new_version=$(uv version | cut -d' ' -f2)

    # Convert Python version to npm-compatible semver
    # Count the dots to determine format
    dot_count=$(echo "$new_version" | tr -cd '.' | wc -c)

    if [ "$dot_count" -eq 1 ]; then
        # Version like "16.0" -> append ".0"
        npm_version="${new_version}.0"
    else
        # Version like "16.0.1" -> use as-is
        npm_version="$new_version"
    fi

    # Set the frontend to the same version
    cd frontend
    npm version "$npm_version" --no-git-tag-version --allow-same-version

    # Commit the version bump
    cd ..
    jj commit -m "$new_version"
