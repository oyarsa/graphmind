"""Remove invalid items from intermediate results. Creates a backup of the original.

The backup has the same path as the original with `.bak` at the end.

The goal to re-run the script with the filtered results file so we attempt the invalid
ones again.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from paper.gpt.evaluate_rationale import GraphWithEval
from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import PaperAnnotated, PromptResult
from paper.util.serde import load_data_jsonl, save_data_jsonl

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def terms(
    input_file: Annotated[Path, typer.Argument(help="Path to terms file")],
) -> None:
    """Remove invalid `PaperAnnotated` entries from intermediate results file.

    Valid terms if items match ALL these criteria:
    - `terms` are valid (see `PaperTerms.is_valid`)
    - `background` is not empty
    - `target` is not empty
    """
    data = load_data_jsonl(input_file, PromptResult[PaperAnnotated])

    empty_terms_count = sum(1 for entry in data if not entry.item.terms.is_valid())
    empty_abstract_count = sum(
        1 for entry in data if not entry.item.background or not entry.item.target
    )

    filtered = [
        entry
        for entry in data
        if entry.item.terms.is_valid() and entry.item.background and entry.item.target
    ]

    print(f"All: {len(data)}.")
    print(f"Empty terms: {empty_terms_count}")
    print(f"Empty abstract: {empty_abstract_count}")
    print(f"Filtered: {len(filtered)}")

    backup_file = input_file.parent / f"{input_file}.bak"
    input_file.rename(backup_file)
    save_data_jsonl(input_file, filtered)


@app.command(help=__doc__, no_args_is_help=True)
def graph(
    input_file: Annotated[Path, typer.Argument(help="Path to graph file")],
) -> None:
    """Remove invalid `GraphResult` entries from intermediate results file.

    An entry is valid if it's not empty. See `GraphResult.is_empty`.
    """
    data = load_data_jsonl(input_file, PromptResult[GraphResult])

    empty_graph_count = sum(1 for entry in data if entry.item.graph.is_empty())

    filtered = [entry for entry in data if not entry.item.graph.is_empty()]

    print(f"All: {len(data)}.")
    print(f"Empty graph: {empty_graph_count}")
    print(f"Filtered: {len(filtered)}")

    copy_backup_and_save(input_file, filtered)


@app.command(help=__doc__, no_args_is_help=True)
def rationale(
    input_file: Annotated[
        Path, typer.Argument(help="Path to rationale evaluation file")
    ],
) -> None:
    """Remove invalid `GraphWithEval` entries from intermediate results file.

    See `GraphWithEval.is_valid`.
    """
    data = load_data_jsonl(input_file, PromptResult[GraphWithEval])

    empty_eval_count = sum(
        1 for entry in data if not entry.item.eval_metrics.is_valid()
    )

    filtered = [entry for entry in data if entry.item.eval_metrics.is_valid()]

    print(f"All: {len(data)}.")
    print(f"Empty graph: {empty_eval_count}")
    print(f"Filtered: {len(filtered)}")

    copy_backup_and_save(input_file, filtered)


def copy_backup_and_save(input_file: Path, filtered: Sequence[BaseModel]) -> None:
    """Copy existing file to `.bak` and write new file with `filtered`. Uses JSONLines."""
    backup_file = input_file.parent / f"{input_file}.bak"
    input_file.rename(backup_file)
    save_data_jsonl(input_file, filtered, mode="w")


if __name__ == "__main__":
    app()
