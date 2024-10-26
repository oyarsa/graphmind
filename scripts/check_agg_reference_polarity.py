"""Check for difference in polarities from references with multiple contexts.

The input file should have the `paper.asap.models.PaperWithReferenceEnriched`
format.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.asap.model import (
    CitationContext,
    PaperWithReferenceEnriched,
)

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
    contexts_classified = [
        context
        for paper in papers
        for reference in paper.references
        for context in reference.contexts
    ]

    references: list[tuple[str, list[CitationContext]]] = []

    for paper in papers:
        for reference in paper.references:
            contexts_classified: list[CitationContext] = []
            for context in reference.contexts:
                if context.polarity is not None:
                    contexts_classified.append(context)

            if len(contexts_classified) > 1:
                references.append((reference.s2title, contexts_classified))

    print(len(references))


if __name__ == "__main__":
    app()
