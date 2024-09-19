#!/usr/bin/env bash

print_usage() {
	echo "Usage: $0 <task>"
	echo "Run '$0 help' for a list of available tasks."
}

print_help() {
	echo "Usage: $0 <task>"
	echo
	echo "Available tasks:"
	echo "  check   Run ruff check, ruff format, and pyright"
	echo "  help   Show this help message"
}

check() {
	uv run ruff check . --fix
	uv run ruff format .
	uv run pyright .
}

if [ $# -eq 0 ]; then
	print_usage
	exit 1
fi

case "$1" in
help)
	print_help
	;;
check)
	check
	;;
*)
	echo "Error: Unknown task '$1'"
	print_usage
	exit 1
	;;
esac
