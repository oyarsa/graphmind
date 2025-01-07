"""Process paper content and review JSON files into a single JSON file.

Only keep those entries where all reviews have ORIGINALITY and REVIEWER_CONFIDENCE
ratings.

The original JSON files had some encoding issues, so this sanitises the text to be valid
UTF-8.

Use the `peerread.download` program to download the data.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import typer
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from paper.peerread.model import (
    CitationContext,
    Paper,
    PaperReference,
    PaperReview,
    PaperSection,
)
from paper.util import groupby
from paper.util.serde import safe_load_json, save_data


class _Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    rationale: str
    rating: int
    confidence: int | None


class _PaperReviews(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    reviews: Sequence[_Review]
    conference: str
    accepted: bool | None


class _PaperMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    title: str
    abstract: str
    authors: Sequence[str]
    sections: Sequence[PaperSection]
    references: Sequence[PaperReference]
    conference: str


def _merge_section(subsections: Sequence[PaperSection]) -> PaperSection | None:
    """Merge subsections into a single section.

    Uses the first subsection's heading as the title. The text is the concatenation of
    all subsections' headings and texts.

    Returns:
        Section with combined titles and texts, or None if subsections is empty.
    """
    if not subsections:
        return None

    title = subsections[0].heading
    text = "\n".join(
        [f"{subsection.heading}\n{subsection.text}\n" for subsection in subsections]
    )
    return PaperSection(heading=title, text=text)


def _parse_section_number(heading: str) -> int | None:
    """Get section number from heading.

    Args:
        heading: Full section heading text.

    Returns:
        Leading heading number if it's numeric, or None if it's a letter heading (e.g.
        appendix).

    Examples:
        >>> parse_section_number("1. Introduction")
        1
        >>> parse_section_number("3.1 Experiment settings")
        3
        >>> parse_section_number("A Appendix")
        None
    """
    section_mark = heading.split(" ")[0]
    section_number = section_mark.split(".")[0].strip()
    try:
        return int(section_number)
    except ValueError:
        return None


def _group_sections(sections: Iterable[dict[str, str]]) -> list[PaperSection]:
    r"""Combine subsections with the same main heading number into a single section.

    Combines the text of each matching subsection into a single block.
    Ignores sections with letter headings (e.g., appendix).

    Args:
        sections: List of sections from a paper as dicts with string keys "heading" and
            "text".

    Returns:
        List of Sections with subsections merged.

    Example:
        >>> sections = [
        ...     {"heading": "1. Introduction", "text": "A"},
        ...     {"heading": "2. Related work", "text": "B1"},
        ...     {"heading": "2.1 Related subsection", "text": "B2"},
        ... ]
        >>> group_sections(sections)
        [
            Section(heading="1. Introduction", text="1. Introduction\nA\n"),
            Section(
                heading="2. Related work",
                text="2. Related Work\n\nB1\n2.1 Related subsection\nB2\n",
            )
        ]
    """
    headings = [
        PaperSection(heading=section["heading"], text=section["text"])
        for section in sections
        if section["heading"]
    ]
    heading_groups = groupby(headings, lambda x: _parse_section_number(x.heading))

    return [
        merged
        for subsections in heading_groups.values()
        if (merged := _merge_section(subsections))
    ]


def _extract_conference(path: Path, segment: str) -> str:
    """Extract conference from `path` and `segment`.

    E.g.: data/PeerRead/iclr_2017/test/reviews/330.json and 'reviews':
    - iclr_2017
    """
    try:
        parts = path.parts
        reviews_idx = parts.index(segment)
        return parts[reviews_idx - 2]
    except ValueError:
        print(path)
        raise


def _process_reviews(directory: Path) -> dict[str, _PaperReviews]:
    """Combine all paper reviews in `directory` in a single array.

    Valid reviews have non-empty ORIGINALITY and REVIEWER_CONFIDENCE fields. Valid papers
    have at least one valid review.

    Metadata is parsed from the file paths and added to the papers: conference, split and
    id.

    Args:
        directory: Path to the directory containing JSON files.

    Returns:
        Map of indexes to processed input papers with valid reviews. See `_get_idx`.
    """
    reviews_valid: list[_PaperReviews] = []

    files = list(directory.rglob("**/reviews/*.json"))
    for file_path in tqdm(files, desc="Reviews"):
        paper_reviews = safe_load_json(file_path)

        filtered_reviews = [
            review
            for review in paper_reviews["reviews"]
            if review.get("ORIGINALITY") is not None
            and review.get("REVIEWER_CONFIDENCE") is not None
        ]
        if not filtered_reviews:
            continue

        reviews_valid.append(
            _PaperReviews(
                id=paper_reviews["id"],
                conference=_extract_conference(file_path, "reviews"),
                accepted=paper_reviews.get("accepted"),
                reviews=[
                    _Review(
                        rationale=review["comments"],
                        rating=review["ORIGINALITY"],
                        confidence=review.get("CONFIDENCE"),
                    )
                    for review in filtered_reviews
                ],
            )
        )

    return {f"{r.conference}.{r.id}": r for r in reviews_valid}


def _process_metadata(
    directory: Path, valid_idxs: set[str]
) -> dict[str, _PaperMetadata]:
    """Combine all paper data (parsed PDFs) in `directory` in a single array.

    Args:
        directory: Path to find parsed PDFs as JSON files.
        valid_idxs: Indices for valid papers. Those not in here will be skipped.

    Returns:
        Map of indexes to processed input papers with valid parsed information.
    """
    metadata_valid: list[_PaperMetadata] = []

    files = list(directory.rglob("**/parsed_pdfs/*.json"))
    for file_path in tqdm(files, desc="Metadata"):
        id_ = int(file_path.stem.split(".")[0])
        conference = _extract_conference(file_path, "parsed_pdfs")
        if _get_idx(conference, id_) not in valid_idxs:
            continue

        data = safe_load_json(file_path)["metadata"]

        if not data["abstractText"] or not data["sections"] or not data["title"]:
            continue

        metadata_valid.append(
            _PaperMetadata(
                id=id_,
                title=data["title"],
                abstract=data["abstractText"],
                authors=data["authors"] or [],
                sections=_group_sections(data["sections"]),
                references=_process_references(data),
                conference=conference,
            )
        )

    return {f"{m.conference}.{m.id}": m for m in metadata_valid}


def _get_idx(conference: str, id_: int) -> str:
    return f"{conference}.{id_}"


def _process_references(paper: dict[str, Any]) -> list[PaperReference]:
    class ReferenceKey(NamedTuple):
        title: str
        authors: Sequence[str]
        year: int

    references = paper["references"]
    references_output: defaultdict[ReferenceKey, set[str]] = defaultdict(set)

    for ref_mention in paper["referenceMentions"]:
        ref_id = ref_mention["referenceID"]

        if not (0 <= ref_id < len(references)):
            continue

        ref_original = references[ref_id]
        ref_author = sorted(ref_original["author"])

        ref_key = ReferenceKey(
            ref_original["title"], tuple(ref_author), ref_original["year"]
        )
        references_output[ref_key].add(ref_mention["context"].strip())

    return [
        PaperReference(
            title=ref.title,
            authors=ref.authors,
            year=ref.year,
            contexts=[
                CitationContext(sentence=context, polarity=None) for context in contexts
            ],
        )
        for ref, contexts in references_output.items()
    ]


def _count_papers(path: Path) -> int:
    return len(list(path.rglob("**/reviews/*.json")))


def _merge_review_metadata(
    reviews_index: dict[str, _PaperReviews], metadata_index: dict[str, _PaperMetadata]
) -> list[Paper]:
    papers: list[Paper] = []

    for idx, review in tqdm(reviews_index.items(), desc="Merge"):
        if idx not in metadata_index:
            continue

        metadata = metadata_index[idx]
        papers.append(
            Paper(
                title=metadata.title,
                abstract=metadata.abstract,
                authors=metadata.authors or [],
                sections=metadata.sections,
                references=metadata.references,
                conference=metadata.conference,
                approval=review.accepted,
                reviews=[
                    PaperReview(
                        rating=r.rating, rationale=r.rationale, confidence=r.confidence
                    )
                    for r in review.reviews
                ],
            )
        )

    return papers


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    path: Annotated[
        Path, typer.Argument(help="Path to directories containing files to merge.")
    ],
    output_file: Annotated[Path, typer.Argument(help="Output merged JSON file.")],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--max-papers", "-n", help="Limit on the number of papers to process."
        ),
    ] = None,
) -> None:
    """Combine PeerRead data from multiple files into a single JSON."""
    papers_count = _count_papers(path)
    if max_papers is not None:
        papers_count = min(max_papers, papers_count)

    reviews = _process_reviews(path)
    metadata = _process_metadata(path, set(reviews))
    papers = _merge_review_metadata(reviews, metadata)[:max_papers]

    print()
    print(f"Papers: {papers_count}")
    print(f"Valid reviews:  {len(reviews)} ({len(reviews)/papers_count:.2%})")
    print(f"Valid metadata: {len(metadata)} ({len(metadata)/papers_count:.2%})")
    print(f"Valid merged:  {len(papers)} ({len(papers)/papers_count:.2%})")

    save_data(output_file, papers)


if __name__ == "__main__":
    app()
