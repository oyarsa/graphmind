"""Group paper sections by heading number and merge subsections."""

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.asap.model import PaperSection


def _merge_section(subsections: Sequence[PaperSection]) -> PaperSection | None:
    """Merge subsections into a single section.

    Uses the first subsection's heading as the title. The text is the concatenation of
    all subsections' headings and texts.

    Returns:
        Section with combined titles and texts, or None if subsections is empty.
    """
    if not subsections:
        return None

    title = subsections[0].heading
    text = "\n".join(
        [f"{subsection.heading}\n{subsection.text}\n" for subsection in subsections]
    )
    return PaperSection(heading=title, text=text)


def _parse_section_number(heading: str) -> int | None:
    """Get section number from heading.

    Args:
        heading: Full section heading text.

    Returns:
        Leading heading number if it's numeric, or None if it's a letter heading (e.g.
        appendix).

    Examples:
        >>> parse_section_number("1. Introduction")
        1
        >>> parse_section_number("3.1 Experiment settings")
        3
        >>> parse_section_number("A Appendix")
        None
    """
    section_mark = heading.split(" ")[0]
    section_number = section_mark.split(".")[0].strip()
    try:
        return int(section_number)
    except ValueError:
        return None


def _groupby[T, K](
    iterable: Iterable[T], key: Callable[[T], K | None]
) -> dict[K, list[T]]:
    """Group items into a dict by key function. Ignores items with None key.

    Args:
        iterable: Iterable of items to group.
        key: Function that takes an element and returns the key to group by. If the
            returned key is None, the item is discarded.

    Returns:
        Dictionary where keys are the result of applying the key function to the items,
        and values are lists of items that share the same. Keeps the original order of
        items in each group. Keys are kept in the same order as they are first seen.
    """
    groups: dict[K, list[T]] = {}

    for item in iterable:
        k = key(item)
        if k is None:
            continue
        if k not in groups:
            groups[k] = []
        groups[k].append(item)

    return groups


def group_sections(sections: Iterable[dict[str, str]]) -> list[PaperSection]:
    r"""Combine subsections with the same main heading number into a single section.

    Combines the text of each matching subsection into a single block.
    Ignores sections with letter headings (e.g., appendix).

    Args:
        sections: List of sections from a paper as dicts with string keys "heading" and
            "text".

    Returns:
        List of Sections with subsections merged.

    Example:
        >>> sections = [
        ...     {"heading": "1. Introduction", "text": "A"},
        ...     {"heading": "2. Related work", "text": "B1"},
        ...     {"heading": "2.1 Related subsection", "text": "B2"},
        ... ]
        >>> group_sections(sections)
        [
            Section(heading="1. Introduction", text="1. Introduction\nA\n"),
            Section(
                heading="2. Related work",
                text="2. Related Work\n\nB1\n2.1 Related subsection\nB2\n",
            )
        ]
    """
    headings = [
        PaperSection(heading=section["heading"], text=section["text"])
        for section in sections
        if section["heading"]
    ]
    heading_groups = _groupby(headings, lambda x: _parse_section_number(x.heading))

    return [
        merged
        for subsections in heading_groups.values()
        if (merged := _merge_section(subsections))
    ]


def process_sections(infile: Path, outfile: Path, limit: int | None = None) -> None:
    """Process sections from the input JSON file and write to the output JSON file.

    Sections are grouped by heading number and merged into a single section. The output
    is grouped by paper, but contains only the sections and nothing else.

    The input file is the output of `paper.asap.merge`.
    """
    data = json.loads(infile.read_text())

    all_outputs = [group_sections(item["paper"]["sections"]) for item in data[:limit]]

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_bytes(
        TypeAdapter(list[list[PaperSection]]).dump_json(all_outputs, indent=2)
    )


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input: Annotated[Path, typer.Argument(help="Path input to JSON file.")],
    output: Annotated[Path, typer.Argument(help="Path output to JSON file.")],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="Limit on the number of source papers to process."
        ),
    ] = None,
) -> None:
    process_sections(input, output, limit)


if __name__ == "__main__":
    app()
