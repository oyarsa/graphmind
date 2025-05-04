#!/usr/bin/env nu
# Show evaluation metrics from JSON files in a pretty table.

# Apply formatting to specified columns in a table (piped in).
def format_table [columns: list<string>]: [table -> table] {
    update cells --columns $columns { |value|
        if ($value | describe) =~ "float|int" {
            $value | into string --decimals 4
        } else {
            $value
        }
    }
}

def main [...files: string] {
    if ($files | is-empty) {
        print --stderr "Usage: show_eval_results.nu FILE1 [FILE2 ...]"
        exit 1
    }

    print $"Git commit: (git rev-parse HEAD)"

    # Show the evaluation metrics for each run
    print "> Evaluation metrics"

    print (
        $files
        | each { |file|
            open $file
            | insert file ($file | path dirname | path basename)
            | select file precision recall f1 accuracy mae mse stats_pred.mean stats_pred.median stats_pred.stdev cost
            | rename --column { stats_pred.mean: mean, stats_pred.median: median, stats_pred.stdev: stdev}
        }
        | sort-by --reverse accuracy
        | format_table ["precision", "recall", "f1", "cost", "stdev"]
    )

    # Show the gold metrics for each run. They should be identical, or something's wrong.
    print "> Gold metrics"

    print (
        $files
        | each { |file|
            open $file
            | insert file ($file | path dirname | path basename)
            | select file accuracy stats_true.mean stats_true.median stats_true.stdev
            | rename --column {stats_true.mean: mean, stats_true.median: median, stats_true.stdev: stdev}
        }
        | sort-by --reverse accuracy
        | format_table ["stdev"]
    )
}
