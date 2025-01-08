"""Count number of context items in data file. Show polarity frequencies, if available.

The input file should have the `paper.peerread.models.PaperWithReferenceEnriched`
format.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.peerread.model import CitationContext
from paper.semantic_scholar.model import PaperWithReferenceEnriched

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
    """Count number of context items in data file. Optonally, show polarity frequencies."""
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

    references: list[tuple[str, list[CitationContext]]] = []
    for paper in papers:
        for reference in paper.references:
            contexts_classified: list[CitationContext] = []
            for context in reference.contexts:
                if context.polarity is not None:
                    contexts_classified.append(context)

            if len(contexts_classified) > 1:
                references.append((reference.s2title, contexts_classified))

    print("References with more than one context:", len(references))


if __name__ == "__main__":
    app()
