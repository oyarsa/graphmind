"""Extract the information we care about from the merged ASAP JSON file.

We extract the title, abstract, full paper sections, ratings, approval decision and
references (including their context in the paper).
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

from paper_hypergraph.asap import process_sections


def _get_introduction(sections: list[dict[str, Any]]) -> str | None:
    """Combine the main introduction and sub-sections into a single introduction.

    Sometimes, the introduction is split into sub-sections, so we need to combine them.
    Assumes that introduction sections start with "1", which seems to almost be the case.

    Returns:
        Combined introduction sections.
        None if no sections starting with "1" are found.
    """
    texts = [
        section["text"].strip()
        for section in sections
        if section["heading"] and section["heading"].startswith("1")
    ]

    if not texts:
        return None

    return "\n\n".join(texts)


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


def _process_references(paper: dict[str, Any]) -> list[dict[str, Any]]:
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
        {
            "title": ref.title,
            "authors": ref.authors,
            "year": ref.year,
            "contexts": list(contexts),
        }
        for ref, contexts in references_output.items()
    ]


def extract_interesting(input_file: Path, output_file: Path) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper_hypergraph.asap.merge`.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, Any]] = []

    for item in data:
        paper = item["paper"]

        introduction = _get_introduction(paper["sections"])
        if not introduction:
            continue

        sections = process_sections.group_sections(paper["sections"])
        if not sections:
            continue

        ratings = [
            r for review in item["review"] if (r := _parse_rating(review["rating"]))
        ]

        output.append(
            {
                "title": paper["title"],
                "abstract": paper["abstractText"],
                "ratings": ratings,
                "sections": [asdict(section) for section in sections],
                "approval": _parse_approval(item["approval"]),
                "references": _process_references(paper),
            }
        )

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    output_file.write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Path to input (filtered) JSON file")
    parser.add_argument("output", type=Path, help="Path to output extracted JSON file")
    args = parser.parse_args()
    extract_interesting(args.input, args.output)


if __name__ == "__main__":
    main()
