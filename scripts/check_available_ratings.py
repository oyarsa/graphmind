"""Check how many ratings other than novelty are available.

Input: `peerread.preprocess` output, `peerread.Paper`.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from paper import peerread as pr
from paper.util.cmd import title
from paper.util.serde import load_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    peer_file: Annotated[
        Path, typer.Option("--peer", help="Path to PeerRead merged file.")
    ],
) -> None:
    """Check ratings from PeerRead papers."""
    peer_data = load_data(peer_file, pr.Paper)

    reviews = [review for paper in peer_data for review in paper.reviews]

    blacklist = {
        "recommendation",
        "is_meta_review",
        "is_annotated",
        "recommendation_unofficial",
    }

    names_all = [
        rating
        for review in reviews
        for rating in review.other_ratings
        if rating not in blacklist
    ]

    print_count(names_all, "All reviews")

    names_main = [
        rating
        for paper in peer_data
        for rating in paper.review.other_ratings
        if rating not in blacklist
    ]
    print_count(names_main, "Main reviews")

    what_many = [
        {k for k in paper.review.other_ratings if k in blacklist} for paper in peer_data
    ]
    title("Cumulative counts of number of ratings")
    for item, count in count_lengths(what_many).items():
        print(f"{item:2}\t{count}")


def print_count[T](values: list[T], desc: str) -> None:
    """Print counter of items in `values`."""
    title(desc)
    max_col = max(len(str(x)) for x in values)
    for name, val in Counter(values).most_common():
        print(f"{name:{max_col}}\t{val}")


def count_lengths(sets: list[set[str]]) -> dict[int, int]:
    """Get cumulative count of sizes of items in the sets. E.g. 1+, 2+, 3..."""
    max_len = max(len(set_) for set_ in sets)
    counts: dict[int, int] = {}
    for length in range(1, max_len + 1):
        counts[length] = sum(len(set_) >= length for set_ in sets)
    return counts


if __name__ == "__main__":
    app()
