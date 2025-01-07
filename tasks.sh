#!/usr/bin/env bash

help() {
	printf "Usage: %s <task>\n" "$0"
	printf "\n"
	printf "Available tasks:\n"
	printf "  lint    Run ruff check, ruff format, and pyright\n"
	printf "  help    Show this help message\n"
	printf "  clidoc  Generate CLI documentation file\n"
}

lint() {
	uv run ruff check . --fix
	uv run ruff format .
	uv run pyright .
}

watch() {
	watchexec --exts=py --clear --restart 'uv run ruff check && uv run pyright'
}

clidoc() {
	for module in gpt peter scimon preprocess; do
		uv run typer paper.$module.cli utils docs --output src/paper/$module/CLI.md &
	done
	wait
}

if [ $# -eq 0 ]; then
	help
	exit
fi

if declare -f "$1" >/dev/null; then
	"$1"
else
	printf "Error: Unknown task '%s'\n\n" "$1"
	help
	exit 1
fi
