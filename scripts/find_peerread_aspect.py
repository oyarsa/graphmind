"""Find the overlap between PeerRead clean papers and ASAP aspect papers."""

from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated

import orjson
import typer
from pydantic import Field, computed_field
from tqdm import tqdm

from paper import peerread as pr
from paper.evaluation_metrics import RatingStats
from paper.types import Immutable
from paper.util.serde import load_data, save_data

AspectLabel = tuple[int, int, str]


class Aspect(Immutable):
    """Represents aspect information of a given paper review.

    Unfortunately, there's no direct connection to what review the aspect information
    corresponds to. Maybe try string matching between `Aspect.text` and
    `PaperReview.rationale`?
    """

    id: Annotated[str, Field(description="ID of the reference paper.")]
    text: Annotated[str, Field(description="Review rationale text.")]
    labels: Annotated[
        list[AspectLabel],
        Field(
            description="List of labels, each a tuple of (start index, end index, label),"
            " where the indices are characters of `text`."
        ),
    ]


class PaperMetadata(Immutable):
    """`.metadata` field of paper contents (relevant fields only)."""

    title: Annotated[str | None, Field(description="Paper title.")]


class PaperContent(Immutable):
    """ASAP paper content files (relevant fields only)."""

    metadata: PaperMetadata


class PaperReviewWithAspect(pr.PaperReview):
    """PeerRead review with ASAP aspect information."""

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

    @computed_field
    @property
    def has_aspect(self) -> bool:
        """True if the paper has at least one matching review with aspect information."""
        return any(r.aspect_labels for r in self.aspect_reviews)


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

    papers_with_aspects = _match_papers_with_aspects(
        peer_titles, asap_title_to_id, asap_aspect_orig, similarity_threshold
    )
    total_matched_reviews = sum(
        sum(bool(r.aspect_labels) for r in p.aspect_reviews)
        for p in papers_with_aspects
    )
    at_least_one_matching_n = sum(p.has_aspect for p in papers_with_aspects)

    print(f"Total matched reviews: {total_matched_reviews=}")
    print(f"Papers with at least one matching aspect: {at_least_one_matching_n=}")

    save_data(output_file, papers_with_aspects)

    without_aspect = [p for p in papers_with_aspects if not p.has_aspect]
    with_aspect = [p for p in papers_with_aspects if p.has_aspect]
    print(f"With aspect: {len(with_aspect)}")
    print(f"Without aspect: {len(without_aspect)}")

    stats_all = RatingStats.calc([p.rating for p in peer_data])
    stats_without_aspect = RatingStats.calc([p.rating for p in without_aspect])
    stats_with_aspect = RatingStats.calc([p.rating for p in with_aspect])
    print(f"\nALL stats:\n{stats_all}")
    print(f"\nWITH aspect stats:\n{stats_without_aspect}")
    print(f"\nWITHOUT aspect stats:\n{stats_with_aspect}")


def _match_papers_with_aspects(
    peer_titles: dict[str, pr.Paper],
    asap_title_to_id: dict[str, str],
    asap_aspect_orig: dict[str, Aspect],
    similarity_threshold: float,
) -> list[PaperWithAspect]:
    """Match PeerRead papers with ASAP aspects and return matched reviews and papers.

    Args:
        peer_titles: Mapping of normalized titles to PeerRead papers.
        asap_title_to_id: Mapping of normalized titles to ASAP paper IDs.
        asap_aspect_orig: Mapping of ASAP paper IDs to their Aspect data.
        similarity_threshold: Minimum similarity score threshold for matching.

    Returns:
        Tuple containing:
        - List of all reviews that were matched with aspects.
        - List of papers that had at least one matching aspect.
        - List of all papers with aspect information (matched or empty).
    """
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

        for i, review in enumerate(paper.reviews):
            if i == review_idx and score >= similarity_threshold:
                aspect_labels = aspect.labels
            else:
                aspect_labels = []

            aspect_reviews.append(
                PaperReviewWithAspect(
                    **review.model_dump(), aspect_labels=aspect_labels
                )
            )

        papers_with_aspects.append(
            PaperWithAspect(**paper.model_dump(), aspect_reviews=aspect_reviews)
        )

    return papers_with_aspects


def _norm_title(title: str) -> str:
    title = "".join(title.split()).strip()
    return title.casefold()


def _build_id_title_map(dir: Path) -> dict[str, str]:
    id_to_title: dict[str, str] = {}

    files = list(dir.rglob("*content*.json"))
    for file in tqdm(files, "ASAP ID to title mapping"):
        data = PaperContent.model_validate(orjson.loads(file.read_bytes()))
        id_ = file.stem.removesuffix("_content")
        if title := data.metadata.title:
            id_to_title[id_] = title

    return id_to_title


if __name__ == "__main__":
    app()
