"""Fix annotated files from the old format to new format.

Old format: separated contexts, contexts_expanded and contexts_annotated.
New format: old a single contexts list with an optional polarity. This has the
`paper.asap.model.PaperWithReferenceEnriched` format.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import TypeAdapter

from paper.asap.model import (
    CitationContext,
    PaperSection,
    PaperWithReferenceEnriched,
    ReferenceEnriched,
)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[Path, typer.Argument(help="Annotated file in old format.")],
    output_file: Annotated[Path, typer.Argument(help="File in the new format.")],
) -> None:
    input_data: list[dict[str, Any]] = json.loads(input_file.read_bytes())

    output_data = [
        PaperWithReferenceEnriched(
            title=paper["title"],
            abstract=paper["abstract"],
            ratings=paper["ratings"],
            sections=[
                PaperSection(heading=section["heading"], text=section["text"])
                for section in paper["sections"]
            ],
            approval=paper["approval"],
            references=[
                ReferenceEnriched(
                    abstract=reference["abstract"],
                    s2title=reference["s2title"],
                    title=reference["title"],
                    year=reference["year"],
                    authors=reference["authors"],
                    contexts=[
                        CitationContext(
                            sentence=citation["regular"], polarity=citation["polarity"]
                        )
                        for citation in reference["contexts_annotated"]
                    ],
                    citation_count=reference["citation_count"],
                    influential_citation_count=reference["influential_citation_count"],
                    reference_count=reference["reference_count"],
                    tldr=reference["tldr"],
                )
                for reference in paper["references"]
            ],
        )
        for paper in input_data
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(
        TypeAdapter(list[PaperWithReferenceEnriched]).dump_json(output_data, indent=2)
    )


if __name__ == "__main__":
    app()
