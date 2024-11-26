"""Combine reference and recommended papers data into a dataset.

The inputs are the ASAP dataset and results from the S2 API:
- asap_final.json: ASAP main papers. Output of `paper.asap.preprocess`.
- papers_recommended.json: S2 recommended papers based on the ASAP papers. Output of
  `external_data.semantic_scholar.recommended`.
- semantic_scholar_final.json: S2 information on papers referenced by the ASAP ones.
  Output of `external_data.semantic_scholar.info`.

This will build two files:
- asap_with_s2_references.json: ASAP papers enriched the (whole) full data of the
  S2 papers.
- asap_related.json: papers related to the input ASAP papers. This includes both
  reference and recommended papers.

TODO: This currently uses asap.S2Paper to represent the S2 output, since that has
`title_query`, which s2.Paper does not have. However, asap.S2Paper doesn't have all
the fields; I need to add them there.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from paper import asap
from paper.external_data import semantic_scholar as s2
from paper.util.serde import load_data, save_data

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
    min_references: Annotated[
        int,
        typer.Option(
            "--min-refs",
            help="Minimum number of matched S2 references for a paper to be added to"
            " the citation-augmented dataset.",
        ),
    ] = 10,
    num_asap: Annotated[
        int | None,
        typer.Option(
            help="How many papers from ASAP will be used to construct the datasets."
            " This applies after S2 matching. We will always use the full S2 and"
            " recommendations data."
        ),
    ] = None,
) -> None:
    asap_papers = load_data(asap_file, asap.Paper)
    reference_papers = load_data(references_file, asap.S2Paper)
    recommended_papers = load_data(recommended_file, s2.PaperRecommended)

    asap_augmented = _augment_asap(asap_papers, reference_papers, min_references)[
        :num_asap
    ]
    s2_references = _unique_asap_refs(asap_augmented)
    recommended_filtered = _filter_recommended(asap_papers, recommended_papers)
    related_papers = s2_references + recommended_filtered

    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(output_dir / "asap_with_s2_references.json", asap_augmented)
    save_data(output_dir / "asap_related.json", related_papers)


def _augment_asap(
    asap_papers: Iterable[asap.Paper],
    s2_papers: Iterable[asap.S2Paper],
    min_references: int,
) -> list[s2.ASAPWithFullS2]:
    """Augment all references in each ASAP paper with their full S2 data.

    Matches S2 and ASAP data by ASAP paper `title` and S2 `title_query`. It's possible
    that some references can't be matched, so we keep only ASAP papers with at least
    `min_references`.
    """
    augmented_papers: list[s2.ASAPWithFullS2] = []
    s2_papers_from_query = {
        s2.clean_title(paper.title_query): paper for paper in s2_papers
    }

    for asap_paper in asap_papers:
        s2_references = [
            s2_paper
            for ref in asap_paper.references
            if (s2_paper := s2_papers_from_query.get(s2.clean_title(ref.title)))
        ]
        if len(s2_references) >= min_references:
            augmented_papers.append(
                s2.ASAPWithFullS2(
                    title=asap_paper.title,
                    abstract=asap_paper.abstract,
                    reviews=asap_paper.reviews,
                    sections=asap_paper.sections,
                    approval=asap_paper.approval,
                    references=s2_references,
                )
            )

    return augmented_papers


def _filter_recommended(
    asap_papers: Iterable[asap.Paper], recommended_papers: Iterable[s2.PaperRecommended]
) -> list[s2.Paper]:
    """Keep only recommended papers from current papers in the ASAP dataset.

    We might not be using all papres in ASAP, so we'll filter out the ones that weren't
    recommended to the current ones.
    """
    asap_titles = {s2.clean_title(paper.title) for paper in asap_papers}
    return [
        rec
        for rec in recommended_papers
        if any(
            s2.clean_title(source)
            for source in rec.sources_asap
            if source in asap_titles
        )
    ]


def _unique_asap_refs(asap_papers: Iterable[s2.ASAPWithFullS2]) -> list[asap.S2Paper]:
    """Get all unique referenced papers from ASAP based on reference's `title_query`."""
    seen_queries: set[str] = set()
    ref_papers: list[asap.S2Paper] = []

    for asap_paper in asap_papers:
        for ref in asap_paper.references:
            ref_title = s2.clean_title(ref.title_query)
            if ref_title in seen_queries:
                continue

            seen_queries.add(ref_title)
            ref_papers.append(ref)

    return ref_papers


if __name__ == "__main__":
    app()
