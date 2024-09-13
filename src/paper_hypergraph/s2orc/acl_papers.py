"""Get matching papers from a list of venues. Saves as one big JSON.GZ file."""

import argparse
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

from paper_hypergraph.s2orc.match_venues import normalise_text


def match_papers(
    venues: list[str], papers: list[dict[str, str]]
) -> Iterator[dict[str, str]]:
    for paper in papers:
        paper_venue = normalise_text(paper["venue"])
        for candidate_venue in venues:
            if candidate_venue in paper_venue:
                yield paper
                break


def main(venues_file: TextIO, papers: list[Path], output_path: Path) -> None:
    venues = [normalise_text(venue) for venue in venues_file]
    print(f"Loaded {len(venues)} venues.")
    output: list[dict[str, str]] = []

    with tqdm(papers) as pbar:
        for paper_file in pbar:
            with gzip.open(paper_file, "rt") as file:
                data = json.load(file)
                matched = match_papers(venues, data)
                output.extend(m | {"source": paper_file.stem} for m in matched)
                pbar.set_postfix(matched=len(output))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as outfile:
        json.dump(output, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "venues_file",
        type=argparse.FileType("r"),
        help="Input file containing venue names",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to gzipped JSON file where matches will be written",
    )
    parser.add_argument(
        "papers",
        nargs="+",
        type=Path,
        help="JSON.GZ files containing papers to be matched against venues",
    )
    args = parser.parse_args()
    main(args.venues_file, args.papers, args.output_file)
