"""Merge papers across multiple areas and files.

Reads JSON files containing research papers grouped by area and generates a new JSON file
where each paper appears once with all areas it was found in. Papers are identified by
their corpus ID to detect duplicates. Only papers with non-empty abstracts are included.

The input files should be the output of `paper.external_data.semantic_scholar_areas.py`.

We keep track of the area queries that resulted in these papers, and also how many of
them appeared across the multiple files. We have multiple files because I noticed
differing counts of papers across areas from different runs, so I also needed to know if
it's worthwhile to keep querying multiple times or now.

Results:
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ File                  ┃ All papers ┃ Unique papers ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ tmp/asap_areas_1.json │ 11392      │ 9216          │
│ tmp/asap_areas_2.json │ 11392      │ 9205          │
│ tmp/asap_areas_3.json │ 11392      │ 9211          │
│ tmp/asap_areas_4.json │ 11392      │ 9218          │
│ tmp/asap_areas_5.json │ 11392      │ 9212          │
└───────────────────────┴────────────┴───────────────┘
┏━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Item         ┃ Value ┃
┡━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total        │ 46062 │
│ Consolidated │ 9334  │
│ Avg. sources │ 5.04  │
│ Avg. areas   │ 1.02  │
└──────────────┴───────┘

TLDR: Most papers only appear in one area, and it's not worth querying multiple times.
I'll still need a post-processing step to gather the papers in the end, since there's
at least _some_ duplication.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter
from rich.console import Console
from rich.table import Table

from paper.util import HelpOnErrorArgumentParser


def main(input_files: list[Path], output_path: Path) -> None:
    """Process multiple JSON files and generate a consolidated output file.

    Takes a list of input JSON file paths and an output path, processes all files,
    and writes the consolidated results to the output path. Only papers with non-empty
    abstracts are included in the output.
    """
    input_data = [
        TypeAdapter(list[AreaPapers]).validate_json(file.read_bytes())
        for file in input_files
    ]
    paper_collections = [
        process_item(file, data) for file, data in zip(input_files, input_data)
    ]
    consolidated_papers = merge_paper_collections(paper_collections)

    console = Console()

    count_table = Table("File", "All papers", "Unique papers")
    for file, data, col in zip(input_files, input_data, paper_collections):
        row = (file, sum(len(entry.papers) for entry in data), len(col))
        count_table.add_row(*map(str, row))
    console.print(count_table)

    areas_num = [len(paper.areas) for paper in consolidated_papers]
    src_num = [len(paper.sources) for paper in consolidated_papers]

    result_table = Table("Item", "Value")
    result_table.add_row("Total", str(sum(len(col) for col in paper_collections)))
    result_table.add_row("Consolidated", str(len(consolidated_papers)))
    result_table.add_row("Avg. sources", f"{sum(src_num) / len(src_num):.2f}")
    result_table.add_row("Avg. areas", f"{sum(areas_num) / len(areas_num):.2f}")
    console.print(result_table)

    output_path.write_bytes(
        TypeAdapter(list[Paper]).dump_json(consolidated_papers, indent=2)
    )


def process_item(file: Path, data: Sequence[AreaPapers]) -> dict[int, Paper]:
    """Process a single JSON file and extract papers with their areas.

    Creates Paper objects for each paper in the file, using the corpus ID as the key
    to identify unique papers. Papers without abstracts are automatically filtered out
    by the Pydantic validation.
    """
    papers: dict[int, Paper] = {}

    for entry in data:
        area = entry.area
        for paper in entry.papers:
            if not paper.abstract:
                continue

            corpus_id = paper.corpus_id
            if corpus_id not in papers:
                papers[corpus_id] = paper

            papers[corpus_id].areas.add(area)
            papers[corpus_id].sources.add(f"{file.name}-{area}")

    return papers


def merge_paper_collections(collections: Iterable[dict[int, Paper]]) -> list[Paper]:
    """Merge multiple paper collections into a single sorted list.

    Takes an iterable of paper dictionaries and merges them, combining areas for
    papers that appear in multiple collections. Returns a list sorted by corpus ID
    for consistent output.
    """
    merged: dict[int, Paper] = {}

    for collection in collections:
        for corpus_id, paper in collection.items():
            if corpus_id in merged:
                merged[corpus_id].areas.update(paper.areas)
                merged[corpus_id].sources.update(paper.sources)
            else:
                merged[corpus_id] = paper

    return sorted(merged.values(), key=lambda p: p.corpus_id)


class Author(BaseModel):
    name: str
    author_id: Annotated[str | None, Field(alias="authorId")]


class Tldr(BaseModel):
    model: str
    text: str | None


class Paper(BaseModel):
    corpus_id: Annotated[int, Field(alias="corpusId")]
    paper_id: Annotated[str, Field(alias="paperId")]
    url: str
    title: str
    abstract: str | None
    year: int
    reference_count: Annotated[int, Field(alias="referenceCount")]
    citation_count: Annotated[int, Field(alias="citationCount")]
    influential_citation_count: Annotated[int, Field(alias="influentialCitationCount")]
    tldr: Tldr | None
    authors: Sequence[Author]
    areas: set[str] = set()
    sources: set[str] = set()


class AreaPapers(BaseModel):
    area: str
    papers: Sequence[Paper]


if __name__ == "__main__":
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument(
        "input_files", nargs="+", type=Path, help="Input JSON files to process"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output JSON file path"
    )

    args = parser.parse_args()
    main(args.input_files, args.output)
