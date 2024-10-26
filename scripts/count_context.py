"""Count number of context items in data file. Show polarity frequencies, if available.

The input file should have the `paper.asap.models.PaperWithReferenceEnriched`
format.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.asap.model import PaperWithReferenceEnriched

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[Path, typer.Argument(help="Input JSON file.")],
) -> None:
    papers = TypeAdapter(list[PaperWithReferenceEnriched]).validate_json(
        input_file.read_text()
    )
    contexts = [
        context
        for paper in papers
        for reference in paper.references
        for context in reference.contexts
    ]

    print(f"{"contexts":<8} : {len(contexts)}")

    if polarities := [p for context in contexts if (p := context.polarity)]:
        print()
        for key, value in Counter(polarities).most_common():
            print(f"{key:<8} : {value}")


if __name__ == "__main__":
    app()
