#!/usr/bin/env fish
# Run experiments with citation context classification across multiple context types,
# prompts and models

if not set -q N
    echo "Error: N environment variable is not set. Please set it before running" \
        " this script."
    exit 1
end

set -l base_output ./output/context-experiment

# Check $CLEAN to see if we clear the existing experiment results. Otherwise, prompt
# user.
if set -q CLEAN
    if test "$CLEAN" = 0
        echo "Skipping removal of existing experiment results."
    else
        echo "Removing existing experiment results..."
        command rm -rf $base_output
    end
else
    read -P "Delete existing experiment results? (y/n): " choice
    if test "$choice" = y
        command rm -rf $base_output
    end
end

set -l prompt_types full simple
set -l context_modes --use-expanded-context --no-use-expanded-context
set -l models gpt-4o-mini gpt-4o

set -l base_cmd uv run gpt context run --ref-limit $N output/asap_context_ann.json

for prompt in $prompt_types
    for context in $context_modes
        for model in $models
            set -l name {$prompt}_{$context}_{$model}

            set_color yellow
            echo "Running with: $name"
            set_color normal

            set -l cmd "$base_cmd $base_output/$name --user-prompt $prompt $context --model $model"
            eval $cmd

            echo
        end
    end
end

uv run (dirname (status filename))/explore_context_experiments.py $base_output
