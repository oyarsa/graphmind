"""Search paper titles in S2ORC dataset by title.

Takes the S2ORC (no need to involve the whole dataset) and a file containing the paper
names to search as a JSON list.
"""

import json
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from paper.util import fuzzy_ratio

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@dataclass(frozen=True)
class PaperMatch:
    title_query: str
    title_s2orc: str
    score: int


@app.command(help=__doc__)
def main(
    s2orc_index_file: Annotated[Path, typer.Argument(help="S2ORC title index")],
    papers_file: Annotated[
        Path, typer.Argument(help="JSON file with papers to search")
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

    matches_exact = _search_papers_exact(processed_query, processed_s2orc)
    print(f"{len(matches_exact)=}")

    matches_fuzzy = _search_papers_fuzzy(processed_query, processed_s2orc, min_fuzzy)
    print(f"{len(matches_fuzzy)=}")


def _search_papers_exact(
    papers_search: set[str], papers_s2orc: set[str]
) -> list[PaperMatch]:
    output: list[PaperMatch] = []

    for paper in tqdm(papers_search, desc="Searching exact matches"):
        if paper in papers_s2orc:
            output.append(PaperMatch(title_query=paper, title_s2orc=paper, score=100))

    return output


def _search_papers_fuzzy(
    papers_search: set[str], papers_s2orc: set[str], min_fuzzy: int
) -> list[PaperMatch]:
    search_func = partial(
        _search_paper_fuzzy, papers_s2orc=papers_s2orc, min_fuzzy=min_fuzzy
    )

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(search_func, papers_search),
                total=len(papers_search),
                desc="Searching fuzzy matches",
            )
        )

    return [r for r in results if r is not None]


def _search_paper_fuzzy(
    query: str, papers_s2orc: set[str], min_fuzzy: int
) -> PaperMatch | None:
    best_score = 0
    best_match = ""
    for s2orc in papers_s2orc:
        score = fuzzy_ratio(query, s2orc)  # output: integer 0-100 where 100 is exact
        if score >= min_fuzzy and score >= best_score:
            best_score = score
            best_match = s2orc

    if best_match:
        return PaperMatch(title_query=query, title_s2orc=best_match, score=best_score)
    return None


def _preprocess_title(title: str) -> str:
    words = title.strip().casefold().split()
    return " ".join(words)


if __name__ == "__main__":
    app()
