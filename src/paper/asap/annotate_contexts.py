"""Annotate citation contexts polarities using both regular and extended contexts.

The user can quit at any time, in that case, the partial annotation will be saved,
and the user can continue annotating by running the script on the previous output file.

Offers two commands:
- annotate: annotate the input data and save to file.
- sample: takes a data file and samples N random contexts from it. These can be from
  multiple papers.
"""

import random
import textwrap
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.asap.model import (
    CitationContext,
    ContextPolarity,
    PaperWithReferenceEnriched,
    ReferenceEnriched,
)
from paper.util import Timer, cli

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command()
def sample(
    input_file: Annotated[
        Path, typer.Argument(help="Path to input JSON file (ASAP with abstracts).")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file (sampled data).")
    ],
    num_samples: Annotated[int, typer.Argument(help="Number of contexts to sample.")],
    seed: Annotated[int, typer.Option(help="Random seed for sampling")] = 0,
) -> None:
    """Sample N citation contexts from file."""
    random.seed(seed)

    input_data = TypeAdapter(list[PaperWithReferenceEnriched]).validate_json(
        input_file.read_bytes()
    )
    total = _count_contexts(input_data)

    indices = random.sample(list(range(total)), k=num_samples)
    cur_idx = 0

    output_data: list[PaperWithReferenceEnriched] = []

    for paper in input_data:
        picked_references: list[ReferenceEnriched] = []
        for r in paper.references:
            picked_context: list[CitationContext] = []

            for context in r.contexts:
                if cur_idx in indices:
                    picked_context.append(context)

                cur_idx += 1

            if picked_context:
                picked_references.append(
                    ReferenceEnriched(
                        # Updated
                        contexts=picked_context,
                        # The rest remains the same
                        title=r.title,
                        year=r.year,
                        authors=r.authors,
                        abstract=r.abstract,
                        s2title=r.s2title,
                        reference_count=r.reference_count,
                        citation_count=r.citation_count,
                        influential_citation_count=r.influential_citation_count,
                        tldr=r.tldr,
                    )
                )

        if picked_references:
            output_data.append(
                PaperWithReferenceEnriched(
                    # Updated
                    references=picked_references,
                    # The rest remains the same
                    title=paper.title,
                    abstract=paper.abstract,
                    reviews=paper.reviews,
                    sections=paper.sections,
                    approval=paper.approval,
                )
            )

    output_file.write_bytes(
        TypeAdapter(list[PaperWithReferenceEnriched]).dump_json(output_data, indent=2)
    )


@app.command()
def annotate(
    input_file: Annotated[Path, typer.Argument(help="Path to input JSON file.")],
    output_file: Annotated[Path, typer.Argument(help="Path to output JSON file.")],
    width: Annotated[int, typer.Option(help="Width to wrap displayed text.")] = 100,
) -> None:
    """Annotation citation contexts polarities."""
    input_data = TypeAdapter(list[PaperWithReferenceEnriched]).validate_json(
        input_file.read_bytes()
    )

    total = _count_contexts(input_data)
    typer.echo(f"Number of contexts in file: {total}")

    with Timer() as timer:
        output_data = _annotate(input_data, total, width)

    annotated_before = _count_contexts_annotated(input_data)
    annotated_after = _count_contexts_annotated(output_data)
    annotated_num = annotated_after - annotated_before

    typer.echo()
    typer.echo(f"Annotated: {annotated_num}")
    typer.echo(f"Time     : {timer.human}")
    typer.echo(f"Average  : {timer.seconds / (annotated_num or 1):.2f}s")

    polarities = [
        context.polarity
        for paper in output_data
        for reference in paper.references
        for context in reference.contexts
    ]

    typer.echo()
    for polarity, count in Counter(polarities).most_common():
        typer.echo(f"{polarity!s:<9}: {count}")

    output_file.write_bytes(
        TypeAdapter(list[PaperWithReferenceEnriched]).dump_json(output_data, indent=2)
    )
    assert len(input_data) == len(
        output_data
    ), "Output length should match input even if annotation was not completed"


def _annotate(
    input_data: Sequence[PaperWithReferenceEnriched], total: int, width: int
) -> list[PaperWithReferenceEnriched]:
    count = 0
    output_data: list[PaperWithReferenceEnriched] = []
    # When False, we'll skip asking the user and just copy the old annotation. This is
    # used when the user pickes the 'q' (quit) option, so we save the partial annotation
    # for later.
    annotating = True

    for paper in input_data:
        new_references: list[ReferenceEnriched] = []
        for r in paper.references:
            new_contexts: list[CitationContext] = []
            for old in r.contexts:
                count += 1

                if annotating:
                    polarity = _annotate_context(count, total, old, width=width)
                    if polarity is None:  # User picked quit
                        annotating = False
                else:
                    polarity = None

                new_context = CitationContext(sentence=old.sentence, polarity=polarity)
                new_contexts.append(new_context)

            new_reference = ReferenceEnriched(
                # Updated
                contexts=new_contexts,
                # The rest remains the same
                title=r.title,
                year=r.year,
                authors=r.authors,
                abstract=r.abstract,
                s2title=r.s2title,
                reference_count=r.reference_count,
                citation_count=r.citation_count,
                influential_citation_count=r.influential_citation_count,
                tldr=r.tldr,
            )
            new_references.append(new_reference)

        new_paper = PaperWithReferenceEnriched(
            # Updated
            references=new_references,
            # The rest remains the same
            title=paper.title,
            abstract=paper.abstract,
            reviews=paper.reviews,
            sections=paper.sections,
            approval=paper.approval,
        )
        output_data.append(new_paper)

    return output_data


def _count_contexts_annotated(data: Iterable[PaperWithReferenceEnriched]) -> int:
    """Count total number of *annotated* contexts available in the papers."""
    return sum(
        context.polarity is not None
        for paper in data
        for reference in paper.references
        for context in reference.contexts
    )


def _count_contexts(data: Iterable[PaperWithReferenceEnriched]) -> int:
    """Count total number of *regular* contexts available in the papers."""
    return sum(
        len(reference.contexts) for paper in data for reference in paper.references
    )


_ANNOTATION_CACHE: dict[str, ContextPolarity] = {}


def _annotate_context(
    idx: int, total: int, context: CitationContext, *, width: int
) -> ContextPolarity | None:
    if polarity := _ANNOTATION_CACHE.get(context.sentence):
        return polarity

    if context.polarity is not None:
        _ANNOTATION_CACHE[context.sentence] = context.polarity
        return context.polarity

    prompt = f"""\

# {idx} / {total}

Context
-------
{_wrap(context.sentence, width=width)}

"""
    typer.echo(prompt)
    with Timer() as timer:
        answer = typer.prompt(
            "What is the polarity of this context?",
            type=cli.choice(["p", "u", "n", "q"]),
        )
    if answer == "q":
        return None

    polarity = {
        "p": ContextPolarity.POSITIVE,
        "u": ContextPolarity.NEUTRAL,
        "n": ContextPolarity.NEGATIVE,
    }[answer]
    _ANNOTATION_CACHE[context.sentence] = polarity

    typer.echo(f"Chosen: {polarity} (took {timer.human})")
    return polarity


def _wrap(text: str, *, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


@app.callback(help=__doc__)
def doc() -> None:
    """Empty callback just for top-level documentation."""
    pass


if __name__ == "__main__":
    app()
