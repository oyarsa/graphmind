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
          mae,
          mse,
          mean: .stats_pred.mean,
          median: .stats_pred.median,
          stdev: .stats_pred.stdev,
          cost,
        }' \
        $file
end | jq -s '. | sort_by(-.accuracy)' \
    | python $script_dir/tools/json_to_table.py --fmt '{:.4f}' precision recall f1 correlation cost stdev

# Show the gold metrics for each run. They should be identical, or something's wrong.
echo
echo "> Gold metrics"
for file in $argv
    jq --arg filename (basename (dirname $file)) \
        '{
          file: $filename,
          accuracy,
          mean: .stats_true.mean,
          median: .stats_true.median,
          stdev: .stats_true.stdev,
        }' \
        $file
end | jq -s '. | sort_by(-.accuracy)' \
    | python $script_dir/tools/json_to_table.py --fmt '{:.4f}' stdev
