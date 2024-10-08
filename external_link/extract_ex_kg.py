"""Extract the external papers from reference_mentions (with context) and match them to paper meta information, e.g., titke.
    target paper: aspa
    context classification: (not yet finish)
    reference papers: from online api (not yet finish)
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import re
import csv
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



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
    # citation_pattern = r"\(([^()]+? et al\.?|[^()]+?),\s*(\d{4})\)"
    # citation_pattern = r"\(([^()]+? et al\.?|[^()]+?),\s*(\d{4})(?:;|\))"
    citation_pattern = r"([A-Za-z]+(?: et al\.)?),?\s*(\d{4})"

    for item in data:
        paper = item["paper"]

        
        #extract the titles of the references
        references = paper["references"]
        titles = []
        paper_tuple = []
        for ref in references:
            cite_author = ref["shortCiteRegEx"]
            cite_year = ref["year"]
            title = ref["title"]
            paper_tuple.append((cite_author,cite_year,title))
            
       #match the reference_mention to paper_meta/tile
        reference_mentions = paper["referenceMentions"]
            
        #enumerate the mentions/context to extract (author, year)
        reference_tuples = []
        for ref_sample in reference_mentions:
            ref_id = ref_sample["referenceID"]
            # print(ref_id)
            context = ref_sample["context"]
            citations = re.findall(citation_pattern, context)
            for author,year in citations:
                title = None
                try:
                    author = author.split("et al")[0].strip()
                except:
                    author = author.strip()
                # Check if there's a matching author-year pair in B
                for author_meta, year_meta, title_meta in paper_tuple:
                    author_meta = author_meta.split("et al")[0] if "et al" in author_meta else author_meta
                    if fuzz.partial_ratio(author_meta, author)>0.85 and year == str(year_meta):
                        title = title_meta  # Found the title, break the loop
                        reference_tuples.append((author,year,context,title))
                        break
                

        paper_title = paper["title"].lower().replace(" ", "")
        csv_file = f"{paper_title}.reference.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write each tuple as a row in the CSV file
            writer.writerows(reference_tuples)
        # introduction = _get_introduction(paper["sections"])
        # if not introduction:
        #     continue


        # ratings = [
        #     r for review in item["review"] if (r := _parse_rating(review["rating"]))
        # ]

        # output.append(
        #     {
        #         "title": paper["title"],
        #         "abstract": paper["abstractText"],
        #         "introduction": introduction,
        #         "ratings": ratings,
        #         "approval": _parse_approval(item["approval"]),
        #         "references_titles": titles,
        #     }
        # )

    # print("no.  input papers:", len(data))
    # print("no. output papers:", len(output), f"({len(output) / len(data):.2%})")

    # output_file.write_text(json.dumps(output, indent=2))


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
