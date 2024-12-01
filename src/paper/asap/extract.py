"""Extract the information we care about from the merged ASAP JSON file.

We extract the title, abstract, full paper sections, ratings, approval decision and
references (including their context in the paper).
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import typer
from pydantic import TypeAdapter

from paper.asap import process_sections
from paper.asap.model import CitationContext, Paper, PaperReference, PaperReview


def _parse_rating(rating: str) -> int | None:
    """Parse rating text into a number (e.g., "8: Accept" -> 8).

    Returns:
        The rating number or None if the rating text cannot be parsed.
    """
    try:
        return int(rating.split(":")[0].strip())
    except ValueError:
        return None


def _parse_approval(approval: str) -> bool:
    """Parse approval text into a bool ("Reject" -> False, everything else -> True)."""
    return approval.strip().lower() != "reject"


def _process_references(paper: dict[str, Any]) -> list[PaperReference]:
    class ReferenceKey(NamedTuple):
        title: str
        authors: Sequence[str]
        year: int

    references = paper["references"]
    references_output: defaultdict[ReferenceKey, set[str]] = defaultdict(set)

    for ref_mention in paper["referenceMentions"]:
        ref_id = ref_mention["referenceID"]

        if not (0 <= ref_id < len(references)):
            continue

        ref_original = references[ref_id]
        ref_author = sorted(ref_original["author"])

        ref_key = ReferenceKey(
            ref_original["title"], tuple(ref_author), ref_original["year"]
        )
        references_output[ref_key].add(ref_mention["context"].strip())

    return [
        PaperReference(
            title=ref.title,
            authors=ref.authors,
            year=ref.year,
            contexts=[
                CitationContext(sentence=context, polarity=None) for context in contexts
            ],
        )
        for ref, contexts in references_output.items()
    ]


def _process_paper(item: dict[str, Any]) -> Paper | None:
    """Process a single paper item."""
    paper = item["paper"]

    sections = process_sections.group_sections(paper["sections"])
    if not sections:
        return None

    reviews = [
        PaperReview(rating=rating, rationale=review["review"])
        for review in item["review"]
        if (rating := _parse_rating(review["rating"]))
    ]

    return Paper(
        title=paper["title"],
        abstract=paper["abstractText"],
        reviews=reviews,
        sections=sections,
        approval=_parse_approval(item["approval"]),
        references=_process_references(paper),
    )


def extract_interesting(input_file: Path, output_file: Path) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper.asap.merge`.
    """
    data: list[dict[str, Any]] = json.loads(input_file.read_text())

    results = [_process_paper(paper) for paper in data]
    results_valid = [res for res in results if res]

    print("no.  input papers:", len(data))
    print(
        "no. output papers:",
        len(results_valid),
        f"({len(results_valid) / len(data):.2%})",
    )

    output_file.write_bytes(TypeAdapter(list[Paper]).dump_json(results_valid, indent=2))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input: Annotated[Path, typer.Argument(help="Path to input (filtered) JSON file.")],
    output: Annotated[Path, typer.Argument(help="Path to output extracted JSON file.")],
) -> None:
    """Extract interesting information from the merged ASAP JSON file."""
    extract_interesting(input, output)


if __name__ == "__main__":
    app()
