"""Find the overlap between PeerRead clean papers and ASAP aspect papers."""

import json
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from paper import peerread as pr
from paper.util.serde import load_data, save_data

AspectLabel = tuple[int, int, str]


class Aspect(BaseModel):
    """Represents aspect information of a given paper review.

    Unfortunately, there's no direct connection to what review the aspect information
    corresponds to. Maybe try string matching between `Aspect.text` and
    `PaperReview.rationale`?
    """

    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(description="ID of the reference paper.")]
    text: Annotated[str, Field(description="Review rationale text.")]
    labels: Annotated[
        list[AspectLabel],
        Field(
            description="List of labels, each a tuple of (start index, end index, label),"
            " where the indices are characters of `text`."
        ),
    ]


class PaperMetadata(BaseModel):
    """`.metadata` field of paper contents (relevant fields only)."""

    model_config = ConfigDict(frozen=True)

    title: Annotated[str | None, Field(description="Paper title.")]


class PaperContent(BaseModel):
    """ASAP paper content files (relevant fields only)."""

    model_config = ConfigDict(frozen=True)

    metadata: PaperMetadata


class PaperReviewWithAspect(pr.PaperReview):
    """PeerRead review with ASAP aspect information."""

    model_config = ConfigDict(frozen=True)

    aspect_labels: Annotated[
        list[AspectLabel],
        Field(
            description="Annotated Aspect labels. If none match, this should be the"
            " empty list."
        ),
    ]


class PaperWithAspect(pr.Paper):
    """PeerRead paper with added aspect information from ASAP."""

    aspect_reviews: list[PaperReviewWithAspect]


def find_matching_review(
    aspect_text: str, reviews: Sequence[pr.PaperReview]
) -> tuple[int, float]:
    """Find index of review that best matches the aspect text and its similarity score."""
    best_idx = -1
    best_score = 0.0

    for i, review in enumerate(reviews):
        score = SequenceMatcher(
            None, aspect_text.lower(), review.rationale.lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


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
        Path, typer.Option("--peer", help="Path to merged PeerRead papers")
    ],
    aspect_file: Annotated[
        Path, typer.Option("--aspect", help="Path to ASAP aspect file")
    ],
    asap_dir: Annotated[
        Path, typer.Option(help="Path to directory with ASAP paper files.")
    ],
    output_file: Annotated[
        Path, typer.Option("--output", help="Path to output PeerRead with labels.")
    ],
    similarity_threshold: Annotated[
        float, typer.Option("--similarity", help="Minimum similarity threshold")
    ] = 0.8,
) -> None:
    """Match ASAP aspect information with PeerRead papers."""
    peer_data = load_data(peer_file, pr.Paper)
    aspect_data = load_data(aspect_file, Aspect)

    # Find all unique titles in the clean PeerRead data and normalise them
    peer_titles = {_norm_title(p.title): p for p in tqdm(peer_data, "PeerRead")}

    # Find all ASAP paper IDs that have any originality aspect labels
    asap_aspect_orig: dict[str, Aspect] = {
        entry.id: entry
        for entry in tqdm(aspect_data, "ASAP originality")
        if any("original" in lb[2].lower() for lb in entry.labels)
    }

    # Build ID to title mapping for all papers in ASAP
    asap_id_to_title = _build_id_title_map(asap_dir)
    asap_title_to_id = {
        _norm_title(title): id_ for id_, title in asap_id_to_title.items()
    }

    # Track matching statistics
    total_matched_reviews: list[PaperReviewWithAspect] = []
    papers_with_matches: list[PaperWithAspect] = []

    # Create enhanced papers with aspects
    papers_with_aspects: list[PaperWithAspect] = []
    for norm_title, paper in tqdm(peer_titles.items(), "Matching aspects"):
        if norm_title not in asap_title_to_id:
            # No matching ASAP paper, create paper with empty aspect reviews
            papers_with_aspects.append(
                PaperWithAspect(
                    **paper.model_dump(),
                    aspect_reviews=[
                        PaperReviewWithAspect(**r.model_dump(), aspect_labels=[])
                        for r in paper.reviews
                    ],
                )
            )
            continue

        # Get aspect data for this paper
        paper_id = asap_title_to_id[norm_title]
        if paper_id not in asap_aspect_orig:
            # No aspect data, create paper with empty aspect reviews
            papers_with_aspects.append(
                PaperWithAspect(
                    **paper.model_dump(),
                    aspect_reviews=[
                        PaperReviewWithAspect(**r.model_dump(), aspect_labels=[])
                        for r in paper.reviews
                    ],
                )
            )
            continue

        aspect = asap_aspect_orig[paper_id]
        review_idx, score = find_matching_review(aspect.text, paper.reviews)

        # Create aspect reviews based on matching
        aspect_reviews: list[PaperReviewWithAspect] = []
        has_match = False

        for i, review in enumerate(paper.reviews):
            if i == review_idx and score >= similarity_threshold:
                review_with_aspect = PaperReviewWithAspect(
                    **review.model_dump(), aspect_labels=aspect.labels
                )
                aspect_reviews.append(review_with_aspect)
                total_matched_reviews.append(review_with_aspect)
                has_match = True
            else:
                aspect_reviews.append(
                    PaperReviewWithAspect(**review.model_dump(), aspect_labels=[])
                )

        paper_with_aspect = PaperWithAspect(
            **paper.model_dump(), aspect_reviews=aspect_reviews
        )
        if has_match:
            papers_with_matches.append(paper_with_aspect)

        papers_with_aspects.append(paper_with_aspect)

    print(f"Total matched reviews: {len(total_matched_reviews)=}")
    print(f"Papers with at least one matching aspect: {len(papers_with_matches)=}")

    save_data(output_file, papers_with_matches)


def _norm_title(title: str) -> str:
    title = "".join(title.split()).strip()
    return title.casefold()


def _build_id_title_map(dir: Path) -> dict[str, str]:
    id_to_title: dict[str, str] = {}

    files = list(dir.rglob("*content*.json"))
    for file in tqdm(files, "ASAP ID to title mapping"):
        data = PaperContent.model_validate(json.loads(file.read_bytes()))
        id_ = file.stem.removesuffix("_content")
        if title := data.metadata.title:
            id_to_title[id_] = title

    return id_to_title


if __name__ == "__main__":
    app()
