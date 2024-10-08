"""Extract references for each paper in ASAP.

This version uses the "referenceID" key in each reference mention, and it seems to work.
I'm not sure whether the "referenceID" is always correct, though.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def extract_references_exact(input_file: Path, output_file: Path) -> None:
    """Extract references from merged ASAP file (see paper_hypergraph.asap.merge).

    Uses the "referenceID" key in each reference mention to get the index to the paper's
    "references" list. Extracts the title, author and year information from there, and
    adds the context from the reference mention.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, Any]] = []

    for item in tqdm(data):
        paper = item["paper"]

        references = paper["references"]
        references_output: list[dict[str, str]] = []

        for ref_mention in paper["referenceMentions"]:
            ref_context = ref_mention["context"]
            ref_id = ref_mention["referenceID"]

            if ref_id >= len(references):
                continue

            ref_original = references[ref_id]
            references_output.append(
                {
                    "title": ref_original["title"],
                    "author": ref_original["author"],
                    "year": ref_original["year"],
                    "context": ref_context,
                }
            )

        output.append(
            {
                "paper_title": paper["title"],
                "references": references_output,
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
    extract_references_exact(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
