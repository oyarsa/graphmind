"""Extract the external paper nodes via different relations.
    target paper: aspa
    reference papers: from online api
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import re

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


def extract_references(input_file: Path, output_file: Path) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper_hypergraph.asap.merge`.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, Any]] = []
    # citation_pattern = r"\(([^)]+? et al\.?)(?:,|\s*\()(\d{4})\)"
    citation_pattern = r"\(([^()]+? et al\.?|[^()]+?),\s*(\d{4})\)"

    for item in data:
        paper = item["paper"]

        reference_mentions = paper["referenceMentions"]
            
        #enumerate the mentions/context to extract (author, year)
        reference_tuples = []
        for ref_sample in reference_mentions:
            ref_id = ref_sample["referenceID"]
            print(ref_id)
            context = ref_sample["context"]
            citations = re.findall(citation_pattern, context)
            for author, year in citations:
                reference_tuples.append((author, year))
        
        #extract the titles of the references
        references = paper["references"]
        titles = []
        for ref in references:
            cite_author = ref["shortCiteRegEx"]
            cite_year = ref["year"]
            title = ref["title"]
            
            

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
                "introduction": introduction,
                "ratings": ratings,
                "sections": [asdict(section) for section in sections],
                "approval": _parse_approval(item["approval"]),
                "references_titles": titles,
            }
        )

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    output_file.write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, help="Path to input (filtered) JSON file")
    parser.add_argument("--output", type=Path, help="Path to output extracted JSON file")
    args = parser.parse_args()
    extract_references(args.input, args.output)


if __name__ == "__main__":
    main()