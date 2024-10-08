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
                        references_output.append(
                            {
                                "author": author,
                                "year": year,
                                "context": context,
                                "title": ref_title,
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
        description="Extract references from papers and output as JSON."
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to input JSON file (asap_merged.json)"
    )
    parser.add_argument("output_file", type=Path, help="Path to output JSON file")
    args = parser.parse_args()
    extract_references(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
