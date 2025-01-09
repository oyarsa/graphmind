"""Combine reference and recommended papers data into a dataset.

The inputs are the PeerRead dataset and results from the S2 API:
- peerread_merged.json: PeerRead main papers. Output of `paper.peerread.process`.
- papers_recommended.json: S2 recommended papers based on the PeerRead papers. Output of
  `semantic_scholar.recommended`.
- semantic_scholar_final.json: S2 information on papers referenced by the PeerRead ones.
  Output of `semantic_scholar.info`.
- peerread_areas.json: Result of searches where queries are ICLR subject areas. Output of
  `semantic_scholar.areas`. Optional.

This will build two files:
- peerread_with_s2_references.json: PeerRead papers enriched the (whole) full data of the
  S2 papers. Type: s2.PaperWithS2Refs.
- peerread_related.json: papers related to the input PeerRead papers. This includes both
  reference and recommended papers. Type: s2.Paper. It's technically a union of other
  types too, but they all fit s2.Paper.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from paper import peerread
from paper import semantic_scholar as s2
from paper.util import display_params
from paper.util.serde import Record, load_data, save_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    peerread_file: Annotated[
        Path,
        typer.Option(
            "--peerread",
            help="File with the regular PeerRead main papers (peerread.Paper).",
        ),
    ],
    references_file: Annotated[
        Path,
        typer.Option(
            "--references",
            help="File with PeerRead reference data (s2.PaperFromPeerRead).",
        ),
    ],
    recommended_file: Annotated[
        Path,
        typer.Option(
            "--recommended", help="File with recommended papers (s2.PaperRecommended)."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", help="Path to directory where all files will be saved."
        ),
    ],
    areas_file: Annotated[
        Path | None,
        typer.Option("--areas", help="File with area search results (s2.PaperArea)."),
    ] = None,
    min_references: Annotated[
        int,
        typer.Option(
            "--min-refs",
            help="Minimum number of matched S2 references for a paper to be added to"
            " the citation-augmented dataset.",
        ),
    ] = 5,
    num_peerread: Annotated[
        int | None,
        typer.Option(
            help="How many papers from PeerRead will be used to construct the datasets."
            " This applies after S2 matching. We will always use the full S2 and"
            " recommendations data."
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option(help="Seed used for `random` when sampling.")
    ] = 0,
) -> None:
    """Combine reference and recommended papers data into a dataset."""
    params = display_params()
    print(params)

    random.seed(seed)

    peerread_papers = load_data(peerread_file, peerread.Paper)
    reference_papers = load_data(references_file, s2.PaperFromPeerRead)
    recommended_papers = load_data(recommended_file, s2.PaperRecommended)
    area_papers = load_data(areas_file, s2.PaperArea) if areas_file else []

    peerread_augmented = _augment_peeread(
        peerread_papers, reference_papers, min_references
    )
    peerread_sampled = (
        random.sample(peerread_augmented, k=num_peerread)
        if num_peerread
        else peerread_augmented
    )
    s2_references = _unique_peerread_refs(peerread_sampled)
    recommended_filtered = _filter_recommended(peerread_sampled, recommended_papers)
    related_papers = _dedup_related(s2_references + recommended_filtered + area_papers)

    print(f"Augmented PeerRead with S2 references: {len(peerread_augmented)}")
    print(f"Sampled PeerRead with S2 references: {len(peerread_sampled)}")
    print(f"S2 references: {len(s2_references)}")
    print(f"Filtered recommended papers: {len(recommended_filtered)}")
    print(f"Area papers: {len(area_papers)}")
    print(f"Unique related papers: {len(related_papers)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(output_dir / "peerread_with_s2_references.json", peerread_sampled)
    save_data(output_dir / "peerread_related.json", related_papers)
    (output_dir / "params.txt").write_text(params)


def _augment_peeread(
    peerread_papers: Iterable[peerread.Paper],
    s2_papers: Iterable[s2.PaperFromPeerRead],
    min_references: int,
) -> list[s2.PaperWithS2Refs]:
    """Augment all references in each PeerRead paper with their full S2 data.

    Matches S2 and PeerRead data by PeerRead paper `title` and S2 `title_peer`. It's
    possible that some references can't be matched, so we keep only PeerRead papers with
    at least `min_references`.
    """
    augmented_papers: list[s2.PaperWithS2Refs] = []
    s2_papers_from_query = {
        s2.clean_title(paper.title_peer): paper for paper in s2_papers
    }

    for peerread_paper in peerread_papers:
        s2_references = [
            s2.S2Reference.from_(s2_paper, contexts=ref.contexts)
            for ref in peerread_paper.references
            if (s2_paper := s2_papers_from_query.get(s2.clean_title(ref.title)))
        ]
        if len(s2_references) >= min_references:
            augmented_papers.append(
                s2.PaperWithS2Refs(
                    title=peerread_paper.title,
                    abstract=peerread_paper.abstract,
                    reviews=peerread_paper.reviews,
                    authors=peerread_paper.authors,
                    sections=peerread_paper.sections,
                    rationale=peerread_paper.rationale,
                    rating=peerread_paper.rating,
                    references=s2_references,
                )
            )

    return augmented_papers


def _filter_recommended(
    peerread_papers: Iterable[s2.PaperWithS2Refs],
    recommended_papers: Iterable[s2.PaperRecommended],
) -> list[s2.Paper]:
    """Keep only recommended papers from current papers in the PeerRead dataset.

    We might not be using all papres in PeerRead, so we'll filter out the ones that w
    eren't recommended to the current ones.
    """
    peerread_titles = {s2.clean_title(paper.title) for paper in peerread_papers}
    return [
        rec
        for rec in recommended_papers
        if any(s2.clean_title(source) in peerread_titles for source in rec.sources_peer)
    ]


def _unique_peerread_refs(
    peerread_papers: Iterable[s2.PaperWithS2Refs],
) -> list[s2.PaperFromPeerRead]:
    """Get all unique referenced papers from PeerRead based on reference's `title_peer`."""
    seen_queries: set[str] = set()
    ref_papers: list[s2.PaperFromPeerRead] = []

    for peerread_paper in peerread_papers:
        for ref in peerread_paper.references:
            ref_title = s2.clean_title(ref.title_peer)
            if ref_title in seen_queries:
                continue

            seen_queries.add(ref_title)
            ref_papers.append(ref)

    return ref_papers


def _dedup_related[T: Record](papers: Sequence[T]) -> Sequence[T]:
    """Deduplicate related papers by their respective IDs.

    It doesn't really care about the concrete type of the paper as long as it has an `id`
    property.
    """
    output: list[T] = []
    seen_ids: set[str] = set()

    for paper in papers:
        if paper.id in seen_ids:
            continue
        seen_ids.add(paper.id)
        output.append(paper)

    return output


if __name__ == "__main__":
    app()
