"""Find and count entries with rationale_pred set to '<error>' in evaluation outputs.

This script supports multiple output file types from evaluation scripts and can:
- Count entries with rationale_pred='<error>'.
- Save filtered results (errors only or non-errors only).
- Automatically detect the appropriate data type for the input file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
from beartype.door import is_bearable
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import PromptResult
from paper.util.cli import die
from paper.util.serde import Compress, get_full_type_name, read_file_bytes, save_data

# Types that have rationale_pred field
type ValidType = (
    PaperResult | GraphResult | PromptResult[PaperResult] | PromptResult[GraphResult]
)


def detect_and_validate(
    data_raw: list[Any],
) -> tuple[type[ValidType], Sequence[ValidType]]:
    """Detect the type and validate all items in the data.

    Args:
        data_raw: Raw data loaded from JSON.

    Returns:
        Tuple of (detected_type, validated_items).

    Raises:
        ValueError: If data is empty or no valid type found.
    """
    if not data_raw:
        raise ValueError("Empty data")

    type_variants = [
        PaperResult,
        GraphResult,
        PromptResult[PaperResult],
        PromptResult[GraphResult],
    ]

    for type_variant in type_variants:
        try:
            validated = [type_variant.model_validate(item) for item in data_raw]
        except ValidationError:
            continue
        else:
            return type_variant, validated

    raise ValueError("Could not detect a valid type with rationale_pred field")


def get_rationale_pred(obj: ValidType) -> str:
    """Extract rationale_pred from an object.

    This handles both direct objects and PromptResult wrapped objects.
    """
    if isinstance(obj, PromptResult):
        obj = obj.item

    return obj.rationale_pred


def partition_by_errors(
    items: Sequence[ValidType],
) -> tuple[list[ValidType], list[ValidType]]:
    """Partition items into error and success lists.

    Args:
        items: Sequence of items to partition.

    Returns:
        Tuple of (error_items, success_items).
    """
    error_items: list[ValidType] = []
    success_items: list[ValidType] = []

    for item in items:
        if get_rationale_pred(item) == "<error>":
            error_items.append(item)
        else:
            success_items.append(item)

    return error_items, success_items


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the evaluation output file to analyse."),
    ],
    save: Annotated[
        Path | None,
        typer.Option(
            "--save",
            help="Save entries with errors to this file.",
        ),
    ] = None,
    filter_out: Annotated[
        Path | None,
        typer.Option(
            "--filter",
            help="Save entries without errors to this file.",
        ),
    ] = None,
    compress: Annotated[
        Compress,
        typer.Option(
            "--compress",
            "-c",
            help="Compression method for output files.",
        ),
    ] = Compress.ZSTD,
) -> None:
    """Count entries with rationale_pred='<error>' in evaluation output files."""
    # Load the data
    try:
        data_raw = orjson.loads(read_file_bytes(input_file))
    except Exception as e:
        die(f"Error loading file: {e}")

    # Ensure it's a list
    if not is_bearable(data_raw, list[Any]):
        die("Input file must contain a JSON array")

    # Detect type and validate
    try:
        detected_type, items = detect_and_validate(data_raw)
    except ValueError as e:
        die(f"Invalid data format: {e}")

    print(f"Detected type: {get_full_type_name(detected_type)}")

    # Partition items into errors and successes
    error_items, success_items = partition_by_errors(items)

    error_count = len(error_items)
    success_count = len(success_items)
    total_count = len(items)

    if total_count == 0:
        print("Warning: No items with rationale_pred found")
        return

    error_percentage = (error_count / total_count) * 100
    success_percentage = (success_count / total_count) * 100

    table = Table()
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    table.add_row("Errors", str(error_count), f"{error_percentage:.1f}%")
    table.add_row("Successes", str(success_count), f"{success_percentage:.1f}%")
    table.add_row("Total", str(total_count), "100.0%")

    Console().print(table)

    # Save filtered results if requested
    if save:
        save_data(save, error_items, compress=compress)
        print(f"Saved {len(error_items)} error entries to {save}")

    if filter_out:
        save_data(filter_out, success_items, compress=compress)
        print(f"Saved {len(success_items)} successful entries to {filter_out}")


if __name__ == "__main__":
    app()
