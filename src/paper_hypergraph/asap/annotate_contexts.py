"""Annotate citation contexts polarities using both regular and extended contexts."""

import textwrap
from collections import Counter
from pathlib import Path
from typing import Annotated

import click
import typer
from pydantic import TypeAdapter

from paper_hypergraph.asap.model import (
    ContextAnnotated,
    ContextPolarity,
    PaperWithReferenceEnriched,
    ReferenceEnriched,
)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[Path, typer.Argument(help="Path to input JSON file.")],
    output_file: Annotated[Path, typer.Argument(help="Path to output JSON file.")],
    width: Annotated[int, typer.Option(help="Width to wrap displayed text.")] = 100,
) -> None:
    input_data = TypeAdapter(list[PaperWithReferenceEnriched]).validate_json(
        input_file.read_bytes()
    )

    output_data: list[PaperWithReferenceEnriched] = []
    count = 0
    total = sum(
        1
        for paper in input_data
        for reference in paper.references
        for _ in reference.contexts_annotated or []
    )
    typer.echo(f"Number of contexts in file: {total}")

    for paper in input_data:
        new_references = []
        for r in paper.references:
            new_contexts_annotated = []
            for regular, expanded, old in zip(
                r.contexts,
                r.contexts_expanded,
                r.contexts_annotated or [None] * len(r.contexts),
            ):
                count += 1
                new_context = ContextAnnotated(
                    regular=regular,
                    expanded=expanded,
                    polarity=_annotate_context(
                        count,
                        total,
                        regular,
                        expanded,
                        old,
                        width=width,
                    ),
                )
                new_contexts_annotated.append(new_context)
            
            new_reference = ReferenceEnriched(
                title=r.title,
                year=r.year,
                authors=r.authors,
                contexts=r.contexts,
                contexts_expanded=r.contexts_expanded,
                contexts_annotated=new_contexts_annotated,
                abstract=r.abstract,
                s2title=r.s2title,
                reference_count=r.reference_count,
                citation_count=r.citation_count,
                influential_citation_count=r.influential_citation_count,
                tldr=r.tldr,
            )
            new_references.append(new_reference)
        
        new_paper = PaperWithReferenceEnriched(
            title=paper.title,
            abstract=paper.abstract,
            ratings=paper.ratings,
            sections=paper.sections,
            approval=paper.approval,
            references=new_references,
        )
        output_data.append(new_paper)

    polarities = [
        context.polarity
        for paper in output_data
        for reference in paper.references
        for context in reference.contexts_annotated or []
    ]
    for polarity, count in Counter(polarities).most_common():
        print(f"{polarity}: {count}")

    output_file.write_bytes(
        TypeAdapter(list[PaperWithReferenceEnriched]).dump_json(output_data, indent=2)
    )


_ANNOTATION_CACHE: dict[str, ContextPolarity] = {}


def _annotate_context(
    idx: int,
    total: int,
    regular: str,
    expanded: str,
    old: ContextAnnotated | None,
    *,
    width: int,
) -> ContextPolarity | None:
    if polarity := _ANNOTATION_CACHE.get(regular):
        return polarity

    if old is not None and old.polarity is not None:
        _ANNOTATION_CACHE[regular] = old.polarity
        return old.polarity

    prompt = f"""\

# {idx} / {total}

Regular
-------
{_wrap(regular, width=width)}


Expanded
--------
{_wrap(expanded,width=width)}

"""
    typer.echo(prompt)
    answer = typer.prompt(
        "What is the polarity of this context?", type=click.Choice(["p", "u", "n", "q"])
    )
    if answer == "q":
        return None

    polarity = {
        "p": ContextPolarity.POSITIVE,
        "u": ContextPolarity.NEUTRAL,
        "n": ContextPolarity.NEGATIVE,
    }[answer]
    typer.echo(f"Chosen: {polarity}")
    _ANNOTATION_CACHE[regular] = polarity

    return polarity


def _wrap(text: str, *, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


if __name__ == "__main__":
    app()
