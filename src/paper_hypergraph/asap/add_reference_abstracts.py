"""Match references from ASAP papers with S2 query results to obtain their abstracts.

The goal is to retrieve the abstract for these references. This is done by querying
the S2 API with the reference titles, taking their best match, and then doing our own
fuzzy matching to ensure the titles are compatible.

Note that this process doesn't ensure perfect matching, but it's the best we can do. By
default, we use a minimum fuzzy ratio of 80%, which includes 90% of the S2 matches. If
no papers from S2 clear that threshold for a given reference, the reference is removed.

The ASAP papers used as input come from the ASAP preprocess pipeline, finishing with
paper_hypergraph.asap.filter (asap_filtered.json).

The S2 results come from paper_hypergraph.s2orc.query_s2. Note that the S2 script also
uses the result of the ASAP pipeline, asap_filtered.json.

Since the S2 script takes a long time to run, it cannot be used in the middle of the
pipeline, so this whole part needs to be run manually. The order is:

- `uv run src/paper_hypergraph/asap/pipeline.py data/asap output`:
    - generates output/asap_filtered.json
- `uv run src/paper_hypergraph/s2orc/query_s2.py output/asap_filtered.json output`:
    - uses `output/asap_filtered.json` to generate `semantic_scholar_filtered.json`
- `uv run src/paper_hypergraph/asap/add_reference_abstracts.py output/asap_filtered.json
  output/semantic_scholar_filtered.json output/asap_with_abstracts.json`:
    - combines the whole thing to generate an ASAP data file including the reference
      abstracts

Diagram for this pipeline:

```
+-----------------------------------+
| asap/pipeline.py data/asap output |
+-----------------------------------+
  |
  |     +---------------------+     +---------------------+     +---------------------+
  +---->| asap_filtered.json  |---->| s2orc/query_s2.py   |---->| semantic_scholar_   |
        +---------------------+     +---------------------+     | filtered.json       |
          |                                                     +---------------------+
          |                                                               |
          |     +---------------------+                                   |
          |     | asap/add_reference_ |                                   |
          +---->| abstracts.py        |<----------------------------------+
                +---------------------+
                  |
                  v
        +---------------------+
        | asap_with_          |
        | abstracts.json      |
        +---------------------+
```
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter
from tqdm import tqdm

from paper_hypergraph.asap.model import (
    ASAPDatasetAdapter,
    PaperReference,
    PaperWithFullReference,
    ReferenceWithAbstract,
    S2Paper,
)
from paper_hypergraph.util import fuzzy_ratio


def _match_paper_external(
    source: PaperReference, external_papers: Sequence[S2Paper], min_score: int
) -> ReferenceWithAbstract | None:
    """Find the external paper whose title best matches the source paper.

    Match is done by fuzzy ratio. The highest ratio is chosen, as long as it's higher
    than `min_score`.

    Returns:
        Best external S2 paper match. None if no papers' ratio is over `min_score`.
    """
    best_external = None
    best_score = min_score

    for external in external_papers:
        ratio = fuzzy_ratio(source.title, external.title_query)
        if ratio >= best_score:
            best_external = external
            best_score = ratio

    if best_external is None:
        return None

    return ReferenceWithAbstract(
        title=source.title,
        year=source.year,
        authors=source.authors,
        contexts=source.contexts,
        abstract=best_external.abstract,
        s2title=best_external.title,
    )


def add_references(
    papers_file: Path,
    external_files: Path,
    output_file: Path,
    min_score: int,
    file_limit: int | None,
) -> None:
    """For each paper, match references with S2 API results to get their abstract.

    Matching is done by fuzzy matching, with a minimum fuzzy score. If no S2 papers
    match a given reference, the reference is removed.
    """
    source_papers = ASAPDatasetAdapter.validate_json(papers_file.read_text())[
        :file_limit
    ]
    external_papers = TypeAdapter(list[S2Paper]).validate_json(
        external_files.read_bytes()
    )

    output = [
        PaperWithFullReference(
            title=paper.title,
            abstract=paper.abstract,
            ratings=paper.ratings,
            sections=paper.sections,
            approval=paper.approval,
            references=[
                match
                for ref in paper.references
                if (match := _match_paper_external(ref, external_papers, min_score))
            ],
        )
        for paper in tqdm(source_papers)
    ]
    output_file.write_bytes(
        TypeAdapter(list[PaperWithFullReference]).dump_json(output, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("papers", type=Path, help="Path to ASAP filtered papers file")
    parser.add_argument("external", type=Path, help="Path to S2 reference papers file")
    parser.add_argument("output", type=Path, help="Path to output JSON file")
    parser.add_argument(
        "--min-score",
        type=int,
        default=80,
        help="Minimum fuzzy score to match an external paper",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Maximum source papers to process"
    )
    args = parser.parse_args()
    add_references(args.papers, args.external, args.output, args.min_score, args.limit)


if __name__ == "__main__":
    main()
