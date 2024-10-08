"""Find all unique paper titles with first author in ASAP and save them to a JSON file."""

import argparse
import json
from pathlib import Path
from typing import TypedDict


class Reference(TypedDict):
    title: str
    author: str | None


def main(input_file: Path, output_file: Path) -> None:
    data = json.loads(input_file.read_text())
    all_titles: dict[str, Reference] = {}

    for entry in data:
        references = entry["paper"]["references"]
        for ref in references:
            title = ref["title"]
            author = ref["author"][0] if ref["author"] else None

            if title not in all_titles:
                all_titles[title] = {"title": title, "author": author}

    print(len(all_titles), "unique titles")
    titles_data = sorted(all_titles.values(), key=lambda x: x["title"])
    output_file.write_text(json.dumps(titles_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to input file (asap_merged.json)"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output file: text file with one paper title per line",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
