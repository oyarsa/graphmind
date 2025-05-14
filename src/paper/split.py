"""Split dataset into train, dev and test."""

import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from paper.util import get_in, get_params, groupby, render_params, sample, setup_logging
from paper.util.cli import die
from paper.util.serde import save_data

logger = logging.getLogger("paper.split")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(no_args_is_help=True)
def split(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to merged ORC data.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for split data files."),
    ],
    train: Annotated[float, typer.Option(help="Ratio or count of the training split.")],
    dev: Annotated[float, typer.Option(help="Ratio or count of dev split.")],
) -> None:
    """Split the dataset into training, dev and test.

    If train and dev are integers, they are treated as straight counts of the output
    splits. If they're floats, they're treated as ratios.

    If they're ratios, the train and dev ratios must sum from 0 to 1.
    If they're counts, they must sum to less than the size of the data.

    In all cases, the remainder is used for the test split.

    Note: the format of the data is irrelevant, as long as it's a JSON array. No
    transformations are applied.
    """
    setup_logging()
    params = get_params()
    logger.info(render_params(params))

    data: list[dict[str, Any]] = json.loads(input_file.read_bytes())

    n = len(data)

    if train.is_integer():
        if not dev.is_integer():
            die("Train is integer, so dev must be too.")
        if not (0 <= train < n):
            die(f"Invalid train count: {train}")
        if not (0 <= dev < n):
            die(f"Invalid train count: {dev}")
        if train + dev >= n:
            die("Train and dev counts must sum to less than size of data")

        train_n = int(train)
        dev_n = int(dev)
    else:  # train is ratio
        if dev.is_integer():
            die("Train is a ratio, so dev must be too.")
        if not (0 <= train < 1):
            die(f"Invalid train ratio: {train}")
        if not (0 <= dev < 1):
            die(f"Invalid train ratio: {dev}")
        if train + dev >= 1:
            die("Train and dev ratio must sum to less than 1")

        train_n = int(n * train)
        dev_n = int(n * dev)

    train_split = data[:train_n]
    dev_split = data[train_n : train_n + dev_n]
    test_split = data[train_n + dev_n :]

    logger.info("Train: %d", len(train_split))
    logger.info("Dev: %d", len(dev_split))
    logger.info("Test: %d", len(test_split))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.json").write_text(json.dumps(train_split))
    (output_dir / "dev.json").write_text(json.dumps(dev_split))
    (output_dir / "test.json").write_text(json.dumps(test_split))
    (output_dir / "params.json").write_text(json.dumps(params))


@app.command(no_args_is_help=True)
def balanced(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to merged ORC data.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file for re-balanced data file."),
    ],
    main_class: Annotated[
        int, typer.Option("--class", help="Which class to base the balance.")
    ],
    key: Annotated[str, typer.Option(help="Key of the class in the object.")],
) -> None:
    """Sample the input file so that it's balanced with the chosen `class."""
    setup_logging()
    params = get_params()
    logger.info(render_params(params))

    data: list[dict[str, Any]] = json.loads(input_file.read_bytes())

    frequencies = _get_frequencies(data, key)
    print("Input frequencies")
    _print_frequencies(frequencies)

    if main_class not in frequencies:
        die(f"Invalid class: {main_class}. Choose from: {frequencies.keys()}")

    main_count = frequencies[main_class]
    output_data: list[dict[str, Any]] = []

    class_items = groupby(data, key=lambda d: _get_paper(d)[key])
    for items in class_items.values():
        output_data.extend(sample(items, main_count))

    print("\nOutput frequencies")
    _print_frequencies(_get_frequencies(output_data, key))

    save_data(output_file, output_data)


def _get_frequencies(data: list[dict[str, Any]], key: str) -> Counter[int]:
    return Counter(_get_paper(d)[key] for d in data)


def _print_frequencies(frequencies: Counter[int]) -> None:
    print("class  count")
    for name, count in sorted(frequencies.items(), key=lambda x: x[0]):
        print(f"{name:5}  {count}")


def _get_paper(item: dict[str, Any]) -> dict[str, Any]:
    paths = [
        "item.paper.paper.paper",  # gpt.PromptResult<gpt.ExtractedGraph>
        "item.paper.paper",  # gpt.PromptResult<gpt.PaperWithRelatedSummary>
        "ann.paper",  # scimon.AnnotatedGraphResult
    ]
    for path in paths:
        if (x := get_in(item, path)) is not None:
            return x

    return item


@app.command(no_args_is_help=True)
def downsample(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input data file.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file for ratio-balanced data."),
    ],
    key: Annotated[str, typer.Option(help="Field name to balance by.")],
    ratios: Annotated[
        str, typer.Option(help="Target ratios as string (e.g., '60/40' or '30/30/40')")
    ],
    count: Annotated[
        int | None, typer.Option(help="Total number of items in output dataset")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for random sample")] = 0,
    filter: Annotated[
        str | None, typer.Option(help="JSON path to property that must be valid.")
    ] = None,
) -> None:
    r"""Balance data according to specified ratios for a given key field.

    The number of components in the ratio must match the number of unique values for the
    key. All ratio components must sum to 100. Each ratio corresponds to the labels in
    sorted order.

    If count is specified, the output dataset will contain exactly that many items
    while maintaining the specified ratios. Otherwise, the output will contain the maximum
    number of items possible while maintaining the ratios.

    If `filter` is given, items with where this property is empty or None are excluded
    from the sampling.

    Example:
        paper split downsample -i input.json -o balanced.json --key approval \
            --ratios "60/40" --filter item.paper.rationale
    """
    random.seed(seed)
    setup_logging()
    params = get_params()
    logger.info(render_params(params))

    ratio_pattern = r"^\d+(/\d+)+$"
    if not re.match(ratio_pattern, ratios):
        die(f"Invalid ratio format: {ratios}. Expected format: '60/40' or '30/30/40'")

    ratio_values = [int(r) for r in ratios.split("/")]
    if sum(ratio_values) != 100:
        die(f"Ratios must sum to 100, got {sum(ratio_values)}")

    data: list[dict[str, Any]] = json.loads(input_file.read_bytes())

    def get_key_value(item: dict[str, Any]) -> Any:
        """Get key from item. Supports both nested paper items and raw ORC output."""
        return _get_paper(item).get(key)

    grouped_data = groupby(data, key=get_key_value)
    unique_values = sorted(grouped_data.keys())

    # Verify number of unique values matches number of ratio components
    if len(unique_values) != len(ratio_values):
        die(
            f"Number of unique values ({len(unique_values)}) does not match "
            f"number of ratio components ({len(ratio_values)})"
        )

    console = Console(file=sys.stderr)

    # Print input distribution
    table_input = Table(title="Input distribution")
    table_input.add_column("Value", style="cyan")
    table_input.add_column("Count", style="magenta", justify="right")
    table_input.add_column("Current %", style="green", justify="right")

    for value in unique_values:
        items = grouped_data[value]
        current_percent = (len(items) / len(data)) * 100
        table_input.add_row(str(value), str(len(items)), f"{current_percent:.1f}%")

    console.print(table_input)

    # Determine the target total count
    if count is not None:
        # User-specified total count
        total_output_count = count
    else:
        # Find the limiting class (the one that would have the fewest items after scaling)
        limiting_ratio = float("inf")
        for i, value in enumerate(unique_values):
            items = grouped_data[value]
            # How many total items can we have if this class represents ratio_values[i]% of
            # the total?
            max_total = (len(items) * 100) / ratio_values[i]
            limiting_ratio = min(limiting_ratio, max_total)

        # Calculate the maximum possible total based on limiting ratio
        total_output_count = 0
        for i in range(len(ratio_values)):
            # Calculate how many items we need from this class
            total_output_count += int((limiting_ratio * ratio_values[i]) / 100)

    # Verify the requested total is feasible
    if count is not None:
        for i, value in enumerate(unique_values):
            # Calculate minimum required items for this class based on the ratio
            min_required = (count * ratio_values[i]) / 100
            available = len(grouped_data[value])
            if available < min_required:
                die(
                    f"Not enough items for class '{value}'. Need {min_required:.1f}"
                    f" but only have {available}. Reduce count or adjust ratios."
                )

    # Calculate target count for each class
    target_counts: dict[Any, int] = {}
    actual_total = 0
    for i, value in enumerate(unique_values):
        # Calculate how many items we need from this class
        target_count = int((total_output_count * ratio_values[i]) / 100)
        target_counts[value] = target_count
        actual_total += target_count

    # Handle rounding issues by adjusting the largest class
    if count is not None and actual_total != count:
        adjustment = count - actual_total
        # Find the largest class to adjust
        largest_class = max(unique_values, key=lambda v: target_counts[v])
        target_counts[largest_class] += adjustment

    output_data: list[dict[str, Any]] = []
    for value, items in grouped_data.items():
        target = target_counts[value]
        if filter:
            valid_items = [item for item in items if get_in(item, filter)]
        else:
            valid_items = items
        output_data.extend(sample(valid_items, target))

    random.shuffle(output_data)

    # Print output distribution
    table_output = Table(title="Output distribution")
    table_output.add_column("Value", style="cyan")
    table_output.add_column("Count", style="magenta", justify="right")
    table_output.add_column("Actual %", style="green", justify="right")

    for value in unique_values:
        count = sum(1 for item in output_data if get_key_value(item) == value)
        actual_percent = (count / len(output_data)) * 100
        table_output.add_row(str(value), str(count), f"{actual_percent:.1f}%")

    console.print(table_output)

    save_data(output_file, output_data)
    console.print(
        f"\nSaved [bold green]{len(output_data)}[/bold green] items to"
        f" [bold blue]{output_file}[/bold blue]"
    )


if __name__ == "__main__":
    app()
