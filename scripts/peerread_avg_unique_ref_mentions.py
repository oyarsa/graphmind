"""Calculate total and average number of unique reference contexts per paper in PeerRead.

Also calculates the number of words in the contexts and estimated number of GPT tokens.

Uses the output of `paper.peerread.merge` as input.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input file (PeerRead merged) to calculate statistics from."
        ),
    ],
) -> None:
    """Calculate total and average number of unique reference contexts per paper in PeerRead."""
    counts_unique: list[int] = []
    all_unique: set[str] = set()

    data = json.loads(input_file.read_text())
    for entry in data:
        unique_contexts: set[str] = {
            r["context"].strip() for r in entry["paper"]["referenceMentions"]
        }
        all_unique.update(unique_contexts)
        counts_unique.append(len(unique_contexts))

    print(f"{len(all_unique):,} total unique mentions")
    print(f"{sum(counts_unique) / len(counts_unique):.2f} average unique mentions")

    num_tokens = sum(len(context.split()) for context in all_unique)
    print(f"{num_tokens:,} total words")
    print(f"{round(num_tokens * 1.5):,} estimated GPT tokens")


if __name__ == "__main__":
    app()
