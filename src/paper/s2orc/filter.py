"""Extract ACL-related papers from gzipped JSON files."""

import gzip
import re
import sys
from pathlib import Path
from typing import Annotated

import orjson
import typer
from tqdm import tqdm

# ACL conference keywords
# fmt: off
_ACL_CONFERENCES = [
    "acl", "association for computational linguistics",
    "aacl", "asia-pacific chapter of the association for computational linguistics",
    "applied natural language processing",
    "computational linguistics",
    "conll", "conference on computational natural language learning",
    "eacl", "european chapter of the association for computational linguistics",
    "emnlp", "empirical methods in natural language processing", "empirical methods in nlp",
    "findings of acl", "findings of the association for computational linguistics",
    "iwslt", "international workshop on spoken language translation",
    "naacl", "north american chapter of the association for computational linguistics",
    "semeval", "semantic evaluation",
    "joint conference on lexical and computational semantics",
    "tacl", "transactions of the association for computational linguistics",
    "wmt", "workshop on machine translation",
    "australasian language technology association",
    "amta", "association for machine translation in the americas",
    "coling", "international conference on computational linguistics",
    "eamt", "european association for machine translation",
    "hlt", "human language technology",
    "ijcnlp", "international joint conference on natural language processing",
    "lilt", "linguistic issues in language technology",
    "lrec", "language resources and evaluation conference",
    "mtsummit", "machine translation summit",
    "muc", "message understanding conference",
    "nejlt", "northern european journal of language technology",
    "paclic", "pacific asia conference on language, information and computation",
    "ranlp", "recent advances in natural language processing",
    "rocling", "rocling conference on computational linguistics",
    "tinlap", "theoretical issues in natural language processing",
    "tipster", "text information processing system"
]
# fmt: on

NORMALISE_REGEX = re.compile(r"[^a-z0-9\s]")


def _normalise_text(text: str) -> str:
    """Remove non-alphanumeric characters and convert to lowercase."""
    return NORMALISE_REGEX.sub("", text.lower()).strip()


def _get_unique_venues(files: list[Path]) -> set[str]:
    """Extract unique ACL-related venues from gzipped JSON files."""
    all_venues: set[str] = set()

    for file_path in tqdm(files, desc="Extracting venues"):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = orjson.loads(f.read())
                all_venues.update(
                    venue.casefold().replace("\n", " ")
                    for item in data
                    if (venue := item.get("venue", "").strip())
                )
        except (orjson.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return all_venues


def _get_acl_venues(venues: set[str]) -> set[str]:
    """Find all matching ACL-related venue names from those that appear in the dataset."""
    venues_candidate_normalised = {_normalise_text(conf) for conf in _ACL_CONFERENCES}
    venues_candidate_regex = [
        re.compile(
            r"\b" + r"\s+".join(re.escape(word) for word in venue_norm.split()) + r"\b"
        )
        for venue_norm in venues_candidate_normalised
    ]
    acl_venues: set[str] = set()

    for venue in venues:
        normalised_venue = _normalise_text(venue)
        for venue_regex in venues_candidate_regex:
            if venue_regex.search(normalised_venue):
                acl_venues.add(venue)
                break

    return acl_venues


def _extract_acl_papers(
    files: list[Path], acl_venues: set[str]
) -> list[dict[str, str]]:
    """Extract papers from ACL-related venues."""
    papers: list[dict[str, str]] = []

    for file_path in tqdm(files, desc="Extracting papers"):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = orjson.loads(f.read())
                for paper in data:
                    if paper.get("venue", "").strip().casefold() in acl_venues:
                        papers.append(paper | {"source": file_path.stem})
        except (orjson.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return papers


def filter_papers(input_directory: Path, output_file: Path) -> None:
    """Keep only ACL-related papers from processed JSON.GZ files.

    The input data is the output of the paper.s2orc.extract module.
    """
    input_files = list(input_directory.rglob("*.json.gz"))
    if not input_files:
        raise ValueError(f"No .json.gz files found in {input_directory}")

    acl_papers = process_acl_papers(input_files)

    save_acl_papers(acl_papers, output_file)
    print(f"{len(acl_papers)} ACL-related papers extracted and saved to {output_file}")


def process_acl_papers(input_files: list[Path]) -> list[dict[str, str]]:
    """Process files to extract ACL-related papers.

    Args:
        input_files: List of JSON.GZ files to process.

    Returns:
        List of ACL-related papers with metadata.
    """
    all_venues = _get_unique_venues(input_files)
    acl_venues = _get_acl_venues(all_venues)
    return _extract_acl_papers(input_files, acl_venues)


def save_acl_papers(papers: list[dict[str, str]], output_file: Path) -> None:
    """Save ACL papers to a gzipped JSON file.

    Args:
        papers: List of papers to save.
        output_file: Path to save the output file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wb") as outfile:
        outfile.write(orjson.dumps(papers))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_directory: Annotated[
        Path, typer.Argument(help="Path to the directory containing data files.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to save the output .json.gz file.")
    ],
) -> None:
    """Extract ACL papers from S2ORC files."""
    filter_papers(input_directory, output_file)


if __name__ == "__main__":
    app()
