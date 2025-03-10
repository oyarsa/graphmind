"""Split dataset into train, dev and test."""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from paper.util import get_params, render_params, setup_logging
from paper.util.cli import die

logger = logging.getLogger("paper.split")

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
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

    (output_dir / "train.json").write_text(json.dumps(train_split))
    (output_dir / "dev.json").write_text(json.dumps(dev_split))
    (output_dir / "test.json").write_text(json.dumps(test_split))
    (output_dir / "params.json").write_text(json.dumps(params))


if __name__ == "__main__":
    app()
