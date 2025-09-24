#!/usr/bin/env fish
# Show evaluation metrics from JSON files in a pretty table.

if test -z "$argv"
    echo "Usage: show_eval_results.fish <file1> <file2> ..."
    exit 1
end

echo "Git commit: $(git rev-parse HEAD)"

set script_dir (dirname (dirname (status filename)))

# Validate input files
set valid_files
for file in $argv
    if not test -f $file
        echo "Warning: File not found: $file" >&2
        continue
    end
    if not jq empty $file >/dev/null 2>&1
        echo "Warning: Invalid JSON in: $file" >&2
        continue
    end
    set valid_files $valid_files $file
end

# Check if we have any valid files
if test (count $valid_files) -eq 0
    echo "Error: No valid files to process" >&2
    exit 1
end

echo "Processing $(count $valid_files) evaluation files"

# Show the evaluation metrics for each run
echo
for file in $valid_files
    jq --arg filename (basename (dirname $file)) \
        '{
          file: $filename,
          precision,
          recall,
          f1,
          accuracy,
          cost,
          confidence,
          correlation,
        }' \
        $file
end | jq -s '. | sort_by(-.accuracy)' \
    | python $script_dir/tools/json_to_table.py \
    --fmt '{:7.4f}' precision recall f1 correlation cost confidence accuracy

# Calculate total cost across all files
set total_cost (for file in $valid_files; jq -r '.cost // 0' $file; end | jq -s 'add')
printf '\nTotal cost: $%.4f\n' $total_cost
