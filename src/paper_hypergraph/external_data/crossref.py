"""Download paper information from the CrossRef API.

Takes the processed output from the ASAP pipeline as input (asap_filtered.json), finds
the unique reference titles and queries them on the Crossref API. This querying is done
by title, where we get the top 10 matches, of which we find the best matching by fuzzy
matching. There's a minimum fuzzy threshold (30 by default).

NB: Crossref has very poor performance, so we focus on using
external_data.semantic_scholar (Semantic Scholar API) instead. This script is likely to
break in the future, as it won't be maintained.
"""

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import habanero  # type: ignore
from tqdm import tqdm

from paper_hypergraph.util import fuzzy_ratio


class CrossrefClient:
    def __init__(self, mailto: str | None = None) -> None:
        self.cr = habanero.Crossref(mailto=mailto or "")  # type: ignore

    def get_papers(self, title: str, fields: list[str], limit: int) -> dict[str, Any]:
        """Fetch top N papers using the title. Defaults to N=10."""
        return self.cr.works(  # type: ignore
            query_title=title, select=fields, limit=limit
        )


def get_best_paper(
    title: str, papers: Iterable[dict[str, Any]], fuzz_threshold: int
) -> dict[str, Any] | None:
    """Find the best paper by title fuzzy ratio. Returns None if there are no matches.

    Some paper entries don't have a title, and are obviously ignored. Also, we need
    abstracts, so even if a paper matches, we ignore it if it doesn't have an abstract.
    """
    best_paper = None
    best_ratio = fuzz_threshold

    for paper in papers:
        if (
            (paper_titles := paper["title"])
            and paper_titles
            and (paper_title := paper_titles[0].strip())
            and paper.get("abstract", "").strip()
        ):
            ratio = fuzzy_ratio(title, paper_title)
            if ratio > best_ratio:
                best_paper = paper
                best_ratio = ratio

    if best_paper:
        return {"fuzz_ratio": best_ratio} | best_paper
    else:
        return None


def download(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    fuzz_threshold: int,
    mailto: str | None,
    paper_limit: int,
) -> None:
    cr = CrossrefClient(mailto=mailto)  # type: ignore

    papers = json.loads(input_file.read_text())
    unique_titles: set[str] = {
        reference["title"] for paper in papers for reference in paper["references"]
    }
    fields = [f for field in fields_str.split(",") if (f := field.lower().strip())]

    output_full: list[dict[str, Any]] = []
    output_best: list[dict[str, Any]] = []

    for title in tqdm(unique_titles):
        meta = {"query": {"title": title}}
        result = cr.get_papers(title, fields, paper_limit)
        output_full.append(meta | result)

        if (papers := result.get("message", {}).get("items")) and (
            best := get_best_paper(title, papers, fuzz_threshold)
        ):
            output_best.append(meta | best)

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "crossref_full.json").write_text(json.dumps(output_full))
    (output_path / "crossref_best.json").write_text(json.dumps(output_best))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=Path, help="Input file (asap_filtered.json)")
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the downloaded information"
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="title,author,published,abstract",
        help="Comma-separated list of fields to retrieve",
    )
    parser.add_argument(
        "--ratio", type=int, default=30, help="Minimum ratio for fuzzy matching"
    )
    parser.add_argument(
        "--mailto",
        type=str,
        default=None,
        help="Email to add to request header. Increases reputation and rate limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of papers returned for each query",
    )
    args = parser.parse_args()

    while True:
        try:
            download(
                args.input_file,
                args.fields,
                args.output_path,
                args.ratio,
                args.mailto,
                args.limit,
            )
            break  # If _download completes without interruption, exit the loop
        except KeyboardInterrupt as e:
            choice = input("\n\nCtrl+C detected. Do you really want to exit? (y/n): ")
            if choice.lower() == "y":
                print(e)
                sys.exit()
            else:
                # The loop will continue, restarting _download
                print("Continuing...\n")


if __name__ == "__main__":
    main()
