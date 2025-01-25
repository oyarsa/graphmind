#!/usr/bin/env fish
# Show evaluation metrics from JSON files in a pretty table.
# json-to-table comes from https://github.com/oyarsa/scripts/tree/master/python

if test -z "$argv"
    echo "Usage: show_eval_results.fish <file1> <file2> ..."
    exit 1
end

for file in $argv
    jq --arg filename (basename (dirname $file)) \
        '{file: $filename} + . | del(.confusion, .correlation, .mode)' $file
end | jq -s '. | sort_by(-.accuracy)' \
    | ,json-to-table --fmt '{:.4f}' precision recall f1 correlation cost
