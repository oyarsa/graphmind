"""Merge data from OpenReview and arXiv to build the final ORC dataset files."""

import contextlib
import datetime as dt
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, overload

import orjson
import typer
from beartype.door import is_bearable
from tqdm import tqdm

from paper import peerread as pr
from paper.orc.download import RATING_KEYS, get_reviews, get_value
from paper.util import groupby
from paper.util.serde import safe_load_json, save_data

logger = logging.getLogger("paper.openreview")


def preprocess(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input", "-i", help="Directory containing the data from all conferences."
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="Path to output JSON file.")
    ],
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers", "-n", help="Number of papers to keep in the output."
        ),
    ] = None,
) -> None:
    """Merge data from all conferences, including reviews and parsed paper content.

    Expects that `input_dir` contains a directory structure like this:

    input_dir
    ├── iclr2024
    │  ├── openreview_arxiv.json
    │  └── parsed
    │     ├── paper1.json
    │     └── paper2.json
    └── iclr2025
       ├── openreview_arxiv.json
       └── parsed
          └── paper3.json

    Where the papers inside `parsed` directories are named after the arXiv title.

    The output is a JSON with an array of pr.Paper.
    """
    if num_papers == 0:
        num_papers = None

    papers_raw = _process_conferences(input_dir)
    papers_processed = [
        _process_paper(paper) for paper in tqdm(papers_raw, "Processing raw papers")
    ]
    papers_valid = [p for p in papers_processed if p]
    papers_dedup = _deduplicate_papers(papers_valid)
    papers_saved = papers_dedup[:num_papers]

    logger.info("Raw papers: %d", len(papers_raw))
    logger.info("Processed papers: %d", len(papers_processed))
    logger.info("Valid papers: %d", len(papers_valid))
    logger.info("Deduplicated papers: %d", len(papers_dedup))
    logger.info("Saving papers: %d.", len(papers_saved))

    save_data(output_file, papers_saved)


def _process_conferences(base_dir: Path) -> list[dict[str, Any]]:
    """Process reviews files and paper contents from conferences in `base_dir`."""
    all_papers: list[dict[str, Any]] = []

    conference_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    for conf_path in conference_dirs:
        conference = conf_path.name
        arxiv_file = conf_path / "openreview_arxiv.json"
        parsed_dir = conf_path / "parsed"

        logger.info(f"Processing {conference}...")

        # Skip if required files/directories don't exist
        if not arxiv_file.exists() or not parsed_dir.exists():
            logger.info(f"Skipping {conference} - missing required files")
            continue

        papers: list[dict[str, Any]] = orjson.loads(arxiv_file.read_bytes())
        # Mapping of paper titles (arXiv) to parsed JSON files
        # Look for both .json and .json.zst files
        json_files = list(parsed_dir.glob("*.json")) + list(
            parsed_dir.glob("*.json.zst")
        )
        title_to_path = {f.stem.removesuffix(".json"): f for f in json_files}

        matched = 0

        for paper in tqdm(papers, desc=f"Processing papers in {conference}"):
            arxiv_title = paper.get("arxiv_title")
            if not arxiv_title:
                continue

            if matched_file := title_to_path.get(arxiv_title):
                paper_content = None
                with contextlib.suppress(Exception):
                    paper_content = safe_load_json(matched_file)
                    matched += 1

                if paper_content is not None:
                    all_papers.append({
                        **paper,
                        "paper_content": paper_content,
                        "conference": conference,
                    })

        logger.info(f"Matched: {matched}. Unmatched: {len(papers) - matched}.")

    return all_papers


def _process_paper(paper_raw: dict[str, Any]) -> pr.Paper | None:
    """Transform a raw paper into a `pr.Paper`.

    Returns None if there are no valid reviews with `RATING_KEYS` or there are
    less than 5 valid references (excludes "Unknown Reference").
    """
    reviews = get_reviews(paper_raw)
    parsed: dict[str, Any] = paper_raw["paper_content"]

    reviews_processed = _process_reviews(reviews)
    references = _process_references(parsed["references"])
    if not reviews_processed or len(references) < 5:
        return None

    sections = [
        pr.PaperSection(heading=s["heading"], text=s["content"])
        for s in parsed["sections"]
    ]
    approval = _find_approval(reviews)

    content: dict[str, Any] = paper_raw["content"]
    abstract = _value(str, content, "abstract", "")
    authors = _value(list, content, "authors", [])
    # Year from creation timestamp (in ms)
    year = dt.datetime.fromtimestamp(paper_raw["cdate"] / 1000, tz=dt.UTC).year

    return pr.Paper(
        title=parsed["title"],
        reviews=reviews_processed,
        abstract=abstract,
        authors=authors,
        sections=sections,
        approval=approval,
        conference=paper_raw["conference"],
        references=references,
        year=year,
    )


def _deduplicate_papers(papers: Iterable[pr.Paper]) -> list[pr.Paper]:
    """Remove paper duplicates by title taking the earliest paper by year."""
    return [
        min(paper_group, key=lambda p: p.year if p.year is not None else float("inf"))
        for paper_group in groupby(papers, key=lambda x: x.title).values()
    ]


def _find_approval(reviews: list[dict[str, Any]]) -> bool | None:
    """Find the review with a decision, if it exists."""
    for reply in reviews:
        content = reply["content"]
        if decision := get_value(content, "decision"):
            return decision.lower() != "reject"

    return None


def _process_references(references: list[dict[str, Any]]) -> list[pr.PaperReference]:
    """Transform a raw reference into a `pr.PaperReference`.

    Ignores references with title "Unknown Reference". It's possible that the output
    list is empty.
    """
    output: list[pr.PaperReference] = []

    for ref in references:
        if ref["title"] == "Unknown Reference":
            continue

        output.append(
            pr.PaperReference(
                title=ref["title"],
                year=ref["year"] or 0,
                authors=ref["authors"],
                contexts=[
                    pr.CitationContext(sentence=sentence, polarity=None)
                    for sentence in ref["citation_contexts"]
                ],
            )
        )

    return output


def _process_reviews(reviews: list[dict[str, Any]]) -> list[pr.PaperReview]:
    """Transform raw reviews into a list of `pr.PaperReview`.

    If a review doesn't contain a valid `contribution`, it's skipped.
    The rationale is a combination of the review's sections: summary, strengths,
    weaknesses, questions and limitations. If none of these are given, the review will
    be empty.
    Other integer ratings will be stored in a separate dictionary.
    """
    output: list[pr.PaperReview] = []

    for review in reviews:
        content = review["content"]

        rating = _max_value(_value(_rating, content, key) for key in RATING_KEYS)
        if rating is None:
            continue

        confidence = _value(_rating, content, "confidence")
        rationale = "\n\n".join(
            f"{key.capitalize()}: {value}"
            for key in [
                "summary",
                "strengths",
                "weaknesses",
                "questions",
                "limitations",
            ]
            if (value := _value(str, content, key))
        )
        other_ratings: dict[str, int] = {
            key: value
            for key, item in content.items()
            if (value := _rating(_nested_value(item)))
        }

        output.append(
            pr.PaperReview(
                rating=rating,
                confidence=confidence,
                rationale=rationale,
                other_ratings=other_ratings,
            )
        )

    return output


def _max_value(values: Iterable[int | None]) -> int | None:
    """Find the maximum value in an iterable, ignoring None values.

    Returns None if the iterable is empty or contains only None values.
    """
    filtered_values = [v for v in values if v is not None]
    return max(filtered_values) if filtered_values else None


@overload
def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: T
) -> T: ...


@overload
def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: None = None
) -> T | None: ...


def _value[T](
    type_: Callable[[Any], T], item: dict[str, Any], key: str, default: T | None = None
) -> T | None:
    """Take the value under `key.value`, as is common with the OpenReview API.

    As the value type is Any, you can use `type_` to make sure the output is of a given
    type, including a custom conversion function.
    """
    value = _nested_value(item.get(key, {}), default)
    if value is None:
        return None
    return type_(value)


def _nested_value(
    value: Any | dict[str, Any], default: Any | None = None
) -> Any | None:
    """If `value` is a dict, gets its nested `value` field. Otherwise, returns as-is."""
    if is_bearable(value, dict[str, Any]):
        return value.get("value", default)
    return value


def _rating(x: Any) -> int | None:
    """Parse a rating from a value.

    If the rating is an int, return it directly. Otherwise, try to extract the rating
    from a string, such as "4 - good" or "1: poor".
    """
    if isinstance(x, int):
        return x

    try:
        fst, _ = re.split(r"[\s\W]+", str(x), maxsplit=1)
        return int(fst)
    except ValueError as e:
        logger.debug("Could not convert rating to int: %s", e)
        return None
