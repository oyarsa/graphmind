"""Search paper titles in S2ORC dataset by title.

Takes the S2ORC (no need to involve the whole dataset) and a file containing the paper
names to search as a JSON list.
"""
# pyright: basic

import json
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import pandas as pd
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
    model_config = ConfigDict(frozen=True)

    title_query: str
    title_s2orc: str
    score: int


class Paper(BaseModel):
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
    s2orc_index: dict[str, str] = json.loads(s2orc_index_file.read_bytes())
    papers: list[str] = json.loads(papers_file.read_bytes())

    processed_s2orc = set([_preprocess_title(t) for t in s2orc_index][:num_s2orc])
    processed_query = set(_preprocess_title(t) for t in papers)

    print(f"{len(processed_s2orc)=}")
    print(f"{len(processed_query)=}")

    matches_fuzzy = _search_papers_fuzzy(processed_query, processed_s2orc, min_fuzzy)
    print()
    print(f"{len(matches_fuzzy)=}")

    scores = pd.Series(m.score for p in matches_fuzzy for m in p.matches)
    print(scores.describe())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(TypeAdapter(list[Paper]).dump_json(matches_fuzzy, indent=2))


def _search_papers_fuzzy(
    papers_search: set[str], papers_s2orc: set[str], min_fuzzy: int
) -> list[Paper]:
    search_func = partial(
        _search_paper_fuzzy, papers_s2orc=papers_s2orc, min_fuzzy=min_fuzzy
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
    query: str, papers_s2orc: set[str], min_fuzzy: int
) -> Paper | None:
    matches: list[PaperMatch] = []

    for s2orc in papers_s2orc:
        score = fuzzy_ratio(query, s2orc)  # output: integer 0-100 where 100 is exact
        if score >= min_fuzzy:
            matches.append(
                PaperMatch(title_query=query, title_s2orc=s2orc, score=score)
            )

    if matches:
        return Paper(query=query, matches=sorted(matches, key=lambda x: x.score))
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


if __name__ == "__main__":
    app()
