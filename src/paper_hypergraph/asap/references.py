"""Extract references from merged ASAP file (see paper_hypergraph.asap.merge).

Uses the "referenceID" key in each reference mention to get the index to the paper's
"references" list. Extracts the title, author and year information from there, and
adds the context from the reference mention.

Groups references with the same title/author/year into a single object with a sorted
list of unique contexts.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def extract_references(input_file: Path, output_file: Path) -> None:
    """Extract paper reference mentions and their contexts.

    Groups mentions to the same paper from different contexts into a single entry.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, Any]] = []

    for item in data:
        paper = item["paper"]

        references = paper["references"]
        references_output: dict[tuple[str, Sequence[str], int], dict[str, Any]] = {}

        for ref_mention in paper["referenceMentions"]:
            ref_id = ref_mention["referenceID"]

            if not (0 <= ref_id < len(references)):
                continue

            ref_original = references[ref_id]
            ref_author = sorted(ref_original["author"])
            ref_key = (
                ref_original["title"],
                tuple(ref_author),
                ref_original["year"],
            )

            if ref_key not in references_output:
                references_output[ref_key] = {
                    "title": ref_original["title"],
                    "author": ref_author,
                    "year": ref_original["year"],
                    "contexts": set(),
                }
            references_output[ref_key]["contexts"].add(ref_mention["context"].strip())

        output.append(
            {
                "paper_title": paper["title"],
                "references": [
                    ref | {"contexts": list(ref["contexts"])}
                    for ref in references_output.values()
                ],
            }
        )

    output_file.write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to input JSON file (asap_merged.json)"
    )
    parser.add_argument("output_file", type=Path, help="Path to output JSON file")
    args = parser.parse_args()
    extract_references(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
