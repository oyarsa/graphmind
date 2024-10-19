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
	printf "  check   Run ruff check, ruff format, and pyright\n"
	printf "  help    Show this help message\n"
	printf "  doc     Open the module documentation on the browser\n"
	printf "  test    Run tests\n"
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

check() {
	uv run ruff check . --fix
	uv run ruff format .
	uv run pyright .
}

doc() {
	uv run pdoc paper_hypergraph --docformat google
}

test() {
	uv run pytest tests
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

case "$1" in
help | setup | check | doc | test)
	"$1"
	;;
*)
	printf "Error: Unknown task '%s'\n\n" "$1"
	usage
	exit 1
	;;
esac
