"""Search paper titles in S2ORC dataset by title.

Takes the S2ORC (no need to involve the whole dataset) and a file containing the paper
names to search as a JSON list.
"""

from __future__ import annotations

import copy
import heapq
import json
import re
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tqdm import tqdm

from paper.util import fuzzy_ratio

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


class PaperMatch(BaseModel):
    """Match between ASAP query title and its S2ORC counterpart with its score."""

    model_config = ConfigDict(frozen=True)

    title_asap: str
    title_s2orc: str
    score: int

    def __lt__(self, other: PaperMatch) -> bool:
        """Compare matches by score. If that's a tie, lexicographically by `title_s2orc`."""
        return (self.score, self.title_s2orc) < (other.score, other.title_s2orc)


class Paper(BaseModel):
    """Paper match with its original title query and the matched S2ORC papers."""

    model_config = ConfigDict(frozen=True)

    query: str
    matches: list[PaperMatch]


@app.command(help=__doc__)
def main(
    s2orc_index_file: Annotated[Path, typer.Argument(help="S2ORC title index")],
    papers_file: Annotated[
        Path, typer.Argument(help="JSON file with papers to search")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Output to JSON file with matches")
    ],
    min_fuzzy: Annotated[int, typer.Option(help="Minimum fuzzy ratio to match")] = 80,
    num_s2orc: Annotated[
        int | None,
        typer.Option(help="Number of papers from S2ORC to use (for testing)."),
    ] = None,
) -> None:
    """Search paper titles in S2ORC dataset by title."""

    s2orc_index: dict[str, str] = json.loads(s2orc_index_file.read_bytes())
    papers: list[str] = json.loads(papers_file.read_bytes())

    processed_s2orc = set([_preprocess_title(t) for t in s2orc_index][:num_s2orc])
    processed_query = {_preprocess_title(t) for t in papers}

    print(f"{len(processed_s2orc)=}")
    print(f"{len(processed_query)=}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_file.with_name(f"{output_file.stem}.tmp.jsonl")
    output_intermediate_file.unlink()

    matches_fuzzy = _search_papers_fuzzy(
        processed_query, processed_s2orc, min_fuzzy, output_intermediate_file
    )
    print()
    print(f"{len(matches_fuzzy)=}")

    scores = [m.score for p in matches_fuzzy for m in p.matches]
    print(describe(scores))

    output_file.write_bytes(TypeAdapter(list[Paper]).dump_json(matches_fuzzy, indent=2))


def describe(values: Iterable[float]) -> str:
    """Get descriptive statistics about numeric iterable."""
    pl.Config.set_tbl_hide_column_data_types(True).set_tbl_hide_dataframe_shape(True)
    return str(pl.Series(values).describe())


def _search_papers_fuzzy(
    papers_search: set[str],
    papers_s2orc: set[str],
    min_fuzzy: int,
    output_intermediate_file: Path,
) -> list[Paper]:
    search_func = partial(
        _search_paper_fuzzy,
        papers_s2orc=papers_s2orc,
        min_fuzzy=min_fuzzy,
        output_intermediate_file=output_intermediate_file,
    )

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(search_func, papers_search),
                total=len(papers_search),
            )
        )

    return [r for r in results if r is not None]


def _search_paper_fuzzy(
    query: str, papers_s2orc: set[str], min_fuzzy: int, output_intermediate_file: Path
) -> Paper | None:
    matches = TopKSet(k=5)

    for s2orc in papers_s2orc:
        score = fuzzy_ratio(query, s2orc)
        if score >= min_fuzzy:
            matches.add(PaperMatch(title_asap=query, title_s2orc=s2orc, score=score))

    if items := matches.items:
        paper = Paper(query=query, matches=items)
        with output_intermediate_file.open("a") as f:
            f.write(paper.model_dump_json() + "\n")
        return paper
    return None


_STOP_WORDS = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for"}


def _preprocess_title(title: str) -> str:
    """Clean and normalise paper titles for better matching.

    Performs:
    - Strips whitespace
    - Converts to lowercase
    - Removes punctuation
    - Normalises whitespace
    - Removes special characters
    """
    text = title.strip().casefold()

    # Remove punctuation and special characters
    # Keep alphanumeric and whitespace, replace everything else with space
    text = re.sub(r"[^\w\s]", " ", text)

    words = text.split()
    words = [w for w in words if w not in _STOP_WORDS]
    return " ".join(words)


class TopKSet:
    """Set that keeps the top K `PaperMatch`es.

    If the collection has less than K items, it accepts any new item. Once K is reached,
    only items that are larger than the smallest of the current items on the list are
    added.

    Items added are also added to a set to make them unique. Note that this means that
    `T` must be hashable.

    Access the items on the list via the `items` property, which returns a new list.
    """

    def __init__(self, *, k: int) -> None:
        """Initialise TopK list.

        Args:
            type_: Type of the elements of the list.
            k: How many items to keep.
        """
        self.k = k
        self.data: list[PaperMatch] = []
        self.seen: set[PaperMatch] = set()

    def add(self, item: PaperMatch) -> None:
        """Add new item to collection, depending on the value of `item`.

        - If we have less than k items, just add it.
        - If the new item is larger than the smallest in the list, replace it.
        - Otherwise, ignore it.
        """
        if item in self.seen:
            return
        self.seen.add(item)

        if len(self.data) < self.k:
            heapq.heappush(self.data, item)
        elif item > self.data[0]:
            heapq.heapreplace(self.data, item)

    @property
    def items(self) -> list[PaperMatch]:
        """Items from the collection in a new list, sorted by descending value.

        Both the list and the items are new, so no modifications will affect the
        collection.

        NB: The list is new by construction, and the items are copied with `deepcopy`.
        """
        return [copy.deepcopy(item) for item in sorted(self.data, reverse=True)]


if __name__ == "__main__":
    app()
