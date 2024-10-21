#!/usr/bin/env bash
# Run experiments with citation context classification across multiple context types,
# prompts, and models

set -e

if [ -z "$N" ]; then
	echo "Error: N environment variable is not set. Set it to run this script."
	exit 1
fi

if [ $# -ge 1 ]; then
	base_output="$1"
	echo "Using provided base directory: $base_output"
else
	base_output="./output/context-tmp"
	echo "Using default base directory: $base_output"
fi

# Check CLEAN to see if we should clear the existing experiment results. Otherwise,
# prompt the user.
if [ -n "$CLEAN" ]; then
	if [ "$CLEAN" = "0" ]; then
		echo "Skipping removal of existing experiment results."
	else
		echo "Removing existing experiment results..."
		rm -rf "$base_output"
	fi
else
	read -rp "Delete existing experiment results? (y/n): " choice
	if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
		echo "Removing existing experiment results..."
		rm -rf "$base_output"
	else
		echo "Skipping removal of existing experiment results."
	fi
fi

prompt_types=(full simple sentence)
context_modes=(--use-expanded-context --no-use-expanded-context)
models=(gpt-4o-mini gpt-4o)

base_cmd=(uv run gpt context run --ref-limit "$N" output/asap_context_ann.json)

echo
for prompt in "${prompt_types[@]}"; do
	for context in "${context_modes[@]}"; do
		for model in "${models[@]}"; do
			name="${prompt}_${context}_${model}"

			echo -e "\e[33mRunning with: $name\e[0m"

			cmd=("${base_cmd[@]}" "$base_output/$name" --user-prompt "$prompt"
				"$context" --model "$model")
			"${cmd[@]}"

			echo
		done
	done
done

uv run "$(dirname "${BASH_SOURCE[0]}")/explore_context_experiments.py" "$base_output"
