"""Find all the unique titles of papers in a JSON file and save them to a text file.

The input JSON file should have the following structure (nested):
- List of objects, each with a "paper" key
- Each "paper" object has a "references" key with a list of objects
- Each "references" object has a "title" key with a string value
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, TypeAdapter


class Reference(BaseModel):
    title: str


class Paper(BaseModel):
    references: Sequence[Reference]


class Data(BaseModel):
    paper: Paper


def main(infile: Path, outfile: Path) -> None:
    data = TypeAdapter(list[Data]).validate_json(infile.read_bytes())
    titles = set(p.title.strip() for d in data for p in d.paper.references)

    print(f"Found {len(titles)} unique titles")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(sorted(titles)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("infile", type=Path, help="Path to the input JSON file")
    parser.add_argument("outfile", type=Path, help="Path to the output JSON file")
    args = parser.parse_args()
    main(args.infile, args.outfile)
