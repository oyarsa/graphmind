"""Extract the information we care about from the merged ASAP JSON file.

We currently care about the title, abstract, and introduction of each paper. More will
be added as needed.
"""

import argparse
import json
from pathlib import Path
from typing import Any


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


def extract_interesting(input_file: Path, output_file: Path) -> None:
    """Extract information from the input JSON file and write to the output JSON file.

    The input file is the output of `paper_hypergraph.asap.merge`.
    """
    data = json.loads(input_file.read_text())

    output: list[dict[str, str]] = []

    for item in data:
        paper = item["paper"]

        introduction = _get_introduction(paper["sections"])
        if not introduction:
            continue

        output.append(
            {
                "title": paper["title"],
                "abstract": paper["abstractText"],
                "introduction": introduction,
            }
        )

    print("no.  input papers:", len(data))
    print("no. output papers:", len(output))

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
