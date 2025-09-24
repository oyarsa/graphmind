#!/usr/bin/env fish
# Show evaluation metrics from JSON files in a pretty table.

if test -z "$argv"
    echo "Usage: show_eval_results.fish <file1> <file2> ..."
    exit 1
end

echo "Git commit: $(git rev-parse HEAD)"

set script_dir (dirname (dirname (status filename)))

# Show the evaluation metrics for each run
echo "> Evaluation metrics"
for file in $argv
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
    | python $script_dir/tools/json_to_table.py --fmt '{:.4f}' precision recall f1 correlation cost
