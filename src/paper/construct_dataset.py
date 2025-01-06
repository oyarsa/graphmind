"""Combine reference and recommended papers data into a dataset.

The inputs are the ASAP dataset and results from the S2 API:
- asap_final.json: ASAP main papers. Output of `paper.asap.preprocess`.
- papers_recommended.json: S2 recommended papers based on the ASAP papers. Output of
  `semantic_scholar.recommended`.
- semantic_scholar_final.json: S2 information on papers referenced by the ASAP ones.
  Output of `semantic_scholar.info`.
- asap_areas.json: Result of searches where queries are ICLR subject areas. Output of
  `semantic_scholar.areas`. Optional.

This will build two files:
- asap_with_s2_references.json: ASAP papers enriched the (whole) full data of the
  S2 papers. Type: asap.PaperWithS2Refs.
- asap_related.json: papers related to the input ASAP papers. This includes both
  reference and recommended papers. Type: s2.Paper. It's technically a union of other
  types too, but they all fit s2.Paper.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from paper import asap
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
    asap_file: Annotated[
        Path,
        typer.Option(
            "--asap", help="File with the regular ASAP main papers (asap.Paper)."
        ),
    ],
    references_file: Annotated[
        Path,
        typer.Option(
            "--references", help="File with ASAP reference data (asap.S2Paper)."
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
    num_asap: Annotated[
        int | None,
        typer.Option(
            help="How many papers from ASAP will be used to construct the datasets."
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

    asap_papers = load_data(asap_file, asap.Paper)
    reference_papers = load_data(references_file, s2.S2Paper)
    recommended_papers = load_data(recommended_file, s2.PaperRecommended)
    area_papers = load_data(areas_file, s2.PaperArea) if areas_file else []

    asap_augmented = _augment_asap(asap_papers, reference_papers, min_references)
    asap_sampled = (
        _balanced_sample(asap_augmented, num_asap) if num_asap else asap_augmented
    )
    s2_references = _unique_asap_refs(asap_sampled)
    recommended_filtered = _filter_recommended(asap_sampled, recommended_papers)
    related_papers = _dedup_related(s2_references + recommended_filtered + area_papers)

    print(f"Augmented ASAP with S2 references: {len(asap_augmented)}")
    print(f"Sampled ASAP with S2 references: {len(asap_sampled)}")
    print(f"S2 references: {len(s2_references)}")
    print(f"Filtered recommended papers: {len(recommended_filtered)}")
    print(f"Area papers: {len(area_papers)}")
    print(f"Unique related papers: {len(related_papers)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(output_dir / "asap_with_s2_references.json", asap_sampled)
    save_data(output_dir / "asap_related.json", related_papers)
    (output_dir / "params.txt").write_text(params)


def _augment_asap(
    asap_papers: Iterable[asap.Paper],
    s2_papers: Iterable[s2.S2Paper],
    min_references: int,
) -> list[s2.PaperWithS2Refs]:
    """Augment all references in each ASAP paper with their full S2 data.

    Matches S2 and ASAP data by ASAP paper `title` and S2 `title_asap`. It's possible
    that some references can't be matched, so we keep only ASAP papers with at least
    `min_references`.
    """
    augmented_papers: list[s2.PaperWithS2Refs] = []
    s2_papers_from_query = {
        s2.clean_title(paper.title_asap): paper for paper in s2_papers
    }

    for asap_paper in asap_papers:
        s2_references = [
            s2.S2Reference.from_(s2_paper, contexts=ref.contexts)
            for ref in asap_paper.references
            if (s2_paper := s2_papers_from_query.get(s2.clean_title(ref.title)))
        ]
        if len(s2_references) >= min_references:
            augmented_papers.append(
                s2.PaperWithS2Refs(
                    title=asap_paper.title,
                    abstract=asap_paper.abstract,
                    reviews=asap_paper.reviews,
                    authors=asap_paper.authors,
                    sections=asap_paper.sections,
                    approval=asap_paper.approval,
                    references=s2_references,
                )
            )

    return augmented_papers


def _balanced_sample(
    data: Sequence[s2.PaperWithS2Refs], n: int
) -> list[s2.PaperWithS2Refs]:
    """Sample balanced entries from approved and rejected.

    The goal is the output will always contain balanced classes. This is necessary
    because the dataset has more approvals than rejections.

    Args:
        data: Papers to draw from. We use `random.sample` for this.
        n: Number of total entries in the output. We do ceil(n/2) for the number of
            entries per class.

    Returns:
        Dataset where the number of approvals and rejections is the same.
    """
    approved = [x for x in data if x.approval]
    rejected = [x for x in data if not x.approval]
    k = math.ceil(n / 2)
    return random.sample(approved, k=k) + random.sample(rejected, k=k)


def _filter_recommended(
    asap_papers: Iterable[s2.PaperWithS2Refs],
    recommended_papers: Iterable[s2.PaperRecommended],
) -> list[s2.Paper]:
    """Keep only recommended papers from current papers in the ASAP dataset.

    We might not be using all papres in ASAP, so we'll filter out the ones that weren't
    recommended to the current ones.
    """
    asap_titles = {s2.clean_title(paper.title) for paper in asap_papers}
    return [
        rec
        for rec in recommended_papers
        if any(s2.clean_title(source) in asap_titles for source in rec.sources_asap)
    ]


def _unique_asap_refs(
    asap_papers: Iterable[s2.PaperWithS2Refs],
) -> list[s2.S2Paper]:
    """Get all unique referenced papers from ASAP based on reference's `title_asap`."""
    seen_queries: set[str] = set()
    ref_papers: list[s2.S2Paper] = []

    for asap_paper in asap_papers:
        for ref in asap_paper.references:
            ref_title = s2.clean_title(ref.title_asap)
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
