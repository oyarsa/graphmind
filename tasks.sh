#!/usr/bin/env bash

UV_MIN_VERSION="0.4.0"

usage() {
	printf "Usage: %s <task>\n" "$0"
	printf "Run '%s help' for a list of available tasks.\n" "$0"
}

help() {
	printf "Usage: %s <task>\n" "$0"
	printf "\n"
	printf "Available tasks:\n"
	printf "  setup   Set up the development environment\n"
	printf "  lint    Run ruff check, ruff format, and pyright\n"
	printf "  help    Show this help message\n"
	printf "  clidoc  Generate CLI documentation file\n"
	printf "  sorc    Run sourcery with auto-fix\n"
}

setup() {
	uv_installed=false
	if ! command -v uv >/dev/null 2>&1; then
		printf "uv is not installed. Installing...\n"
		curl -LsSf https://astral.sh/uv/install.sh | sh
		export PATH="$HOME/.cargo/bin:$PATH"
		uv_installed=true
	else
		# Check if uv is at least the minimum version, and upgrade if necessary
		installed_version=$(uv --version | awk '{print $2}')
		if [ "$(printf '%s\n' "$UV_MIN_VERSION" "$installed_version" | sort -V | head -n1)" != \
			"$UV_MIN_VERSION" ]; then
			printf "Installed uv is too old. Updating uv...\n"
			uv self update
		fi
	fi

	printf "\nInstalling dependencies with uv...\n"
	uv sync

	printf "\nInstalling pre-commit hooks...\n"
	uv run pre-commit install

	printf "\nSetup complete. See README.md and CONTRIBUTING.md for more information.\n"

	if [ "$uv_installed" = true ]; then
		printf "\nIMPORTANT: Please restart your shell to use the newly installed uv.\n"
	fi
}

lint() {
	uv run ruff check . --fix
	uv run ruff format .
	uv run pyright .
}

watch() {
	watchexec --exts=py --restart 'uv run ruff check && uv run pyright'
}

clidoc() {
	uv run typer paper.gpt.cli utils docs --output src/paper/gpt/CLI.md
}

sorc() {
	uv run sourcery review . --fix
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

if declare -f "$1" >/dev/null; then
	"$1"
else
	printf "Error: Unknown task '%s'\n\n" "$1"
	usage
	exit 1
fi
