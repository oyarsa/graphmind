#!/usr/bin/env bash

UV_MIN_VERSION="0.4.0"

usage() {
	echo "Usage: $0 <task>"
	echo "Run '$0 help' for a list of available tasks."
}

help() {
	echo "Usage: $0 <task>"
	echo
	echo "Available tasks:"
	echo "  setup   Set up the development environment"
	echo "  check   Run ruff check, ruff format, and pyright"
	echo "  help    Show this help message"
}

setup() {
	if ! command -v uv >/dev/null 2>&1; then
		echo "uv is not installed. Installing..."
		curl -LsSf https://astral.sh/uv/install.sh | sh
	else
		# Check if uv is at least the minimum version, and upgrade if necessary
		installed_version=$(uv --version | awk '{print $2}')
		if [ "$(printf '%s\n' "$UV_MIN_VERSION" "$installed_version" | sort -V | head -n1)" != \
			"$UV_MIN_VERSION" ]; then
			echo "Installed uv is too old. Upgrading uv..."
			uv self update
		fi
	fi

	echo "Installing dependencies with uv..."
	uv sync

	echo "Installing pre-commit hooks..."
	uv run pre-commit install

	echo "Setup complete. Use \"uv run ...\" to run scripts, or activate the venv in .venv"
}

check() {
	uv run ruff check . --fix
	uv run ruff format .
	uv run pyright .
}

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

case "$1" in
help | setup | check)
	"$1"
	;;
*)
	echo "Error: Unknown task '$1'"
	usage
	exit 1
	;;
esac
