"""Show statistics from context classification results.

The same things are shown when the run is done, but I want to be able to generate that
view again.
"""

from pathlib import Path
from typing import Annotated

import typer

from paper.gpt.classify_contexts import PaperWithContextClassfied, show_classified_stats
from paper.gpt.run_gpt import PromptResult
from paper.util.serde import load_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the output file from context classification."),
    ],
) -> None:
    """Show statistics from context classification results."""
    input_data = load_data(input_file, PromptResult[PaperWithContextClassfied])

    papers = [result.item for result in input_data]
    print(f"Papers: {len(papers)}")

    info, _ = show_classified_stats(papers)
    print(info)


if __name__ == "__main__":
    app()
