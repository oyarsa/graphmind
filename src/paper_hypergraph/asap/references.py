"""Extract references for each paper in ASAP.

This version uses fuzzy matching between the author and year in the context and the
papers in the paper reference. Author and year are extracted from the context using
a regex and the author is matched fuzzily, making this inherently noisy.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from thefuzz import fuzz  # type: ignore
from tqdm import tqdm


def _partial_fuzz_ratio(s1: str, s2: str) -> int:
    return fuzz.partial_ratio(s1, s2)  # type: ignore


def extract_references(input_file: Path, output_file: Path) -> None:
    """Extract references from merged ASAP file (see paper_hypergraph.asap.merge).

    Extracts the author and year from the context using a regular expression, matches
    against the papers in the references by fuzzy matching the author and comparing the
    year.

    The output uses the title from the reference, the author and year extracted from
    the reference context, and the full context itself.
    """
    data = json.loads(input_file.read_text())
    citation_regex = re.compile(r"([A-Za-z]+(?: et al\.)?),?\s*(\d{4})")

    output: list[dict[str, Any]] = []

    for item in tqdm(data):
        paper = item["paper"]

        references = [
            (ref["shortCiteRegEx"], ref["year"], ref["title"])
            for ref in paper["references"]
        ]
        references_output: list[dict[str, str]] = []

        for ref_sample in paper["referenceMentions"]:
            context = ref_sample["context"]
            citations = citation_regex.findall(context)

            for author_, year in citations:
                author = author_.split("et al")[0].strip()
                for ref_author_, ref_year, ref_title in references:
                    if not ref_author_:
                        continue

                    author_meta = ref_author_.split("et al")[0].strip()
                    if _partial_fuzz_ratio(author_meta, author) > 85 and year == str(
                        ref_year
                    ):
                        # Get "exact" direct reference mentions to compare with regex
                        # and fuzzy approach.
                        reference_exact_id = ref_sample["referenceID"]
                        reference_exact = paper["references"][reference_exact_id]

                        references_output.append(
                            {
                                "author": author,
                                "year": year,
                                "context": context,
                                "title": ref_title,
                                "reference_id": reference_exact_id,
                                "title_other": reference_exact["title"],
                            }
                        )
                        break

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
    extract_references(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
