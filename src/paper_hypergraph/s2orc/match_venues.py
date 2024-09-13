"""Match conference names from input file and write matches to output file."""

import argparse
import re
from pathlib import Path

# fmt: off
ACL_CONFERENCES = [
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


normalise_re = re.compile(r"[^a-z0-9\s]")


def normalise_text(text: str) -> str:
    """Remove non-alphanumeric characters and convert to lowercase."""
    return normalise_re.sub("", text.lower()).strip()


def match_venues(input_path: Path, output_path: Path) -> None:
    venues_candidate_normalised = {normalise_text(conf) for conf in ACL_CONFERENCES}
    venues_candidate_regex = [
        re.compile(
            r"\b" + r"\s+".join(re.escape(word) for word in norm_conf.split()) + r"\b"
        )
        for norm_conf in venues_candidate_normalised
    ]

    # Real venue names to match against the normalized ones
    venues_real = (
        venue
        for line in input_path.read_text().splitlines()
        if (venue := normalise_text(line))
    )
    venues_found: set[str] = set()

    for venue in venues_real:
        for venue_regex in venues_candidate_regex:
            if venue_regex.search(venue):
                venues_found.add(venue)
                break

    output_path.write_text("\n".join(sorted(venues_found)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input file containing conference names",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output file where matches will be written",
    )

    args = parser.parse_args()
    match_venues(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
