"""Extract only the data relevant for the demonstration tool.

Includes only entries with a valid rationale, accurate novelty label and non-empty
related papers.

Input format: `gpt.PromptResult[gpt.GraphResult]`.
Output format: `DemoPaper` (in this file).
"""

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import Field

from paper import gpt
from paper import peerread as pr
from paper.gpt.model import is_rationale_valid
from paper.types import Immutable, PaperProtocol
from paper.util import sample
from paper.util.serde import Compress, load_data, save_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


def _get_year(paper: PaperProtocol) -> int | None:
    """Get year from paper. If the year is None, try to parse it from the conference."""
    if paper.year is not None:
        return paper.year

    conf_year = paper.conference[-4:]
    try:
        return int(conf_year)
    except ValueError:
        return None


def _clean_conference(conference: str) -> str:
    """Get conference name, excluding the year if present."""
    if len(conference) >= 4 and conference[-4:].isdigit():
        return conference[:-4]
    else:
        return conference


class DemoPaper(Immutable):
    """Paper information."""

    title: Annotated[str, Field(description="Paper title")]
    abstract: Annotated[str, Field(description="Abstract text")]
    authors: Annotated[Sequence[str], Field(description="Names of the authors")]
    sections: Annotated[
        Sequence[pr.PaperSection], Field(description="Sections in the paper text")
    ]
    approval: Annotated[
        bool | None,
        Field(description="Approval decision - whether the paper was approved"),
    ]
    conference: Annotated[
        str, Field(description="Conference where the paper was published")
    ]
    rating: Annotated[int, Field(description="Novelty rating")]
    year: Annotated[int | None, Field(description="Paper publication year")] = None
    id: Annotated[
        str,
        Field(description="Unique ID for the paper based on the title and abstract."),
    ]

    y_true: Annotated[int, Field(description="Human annotation")]
    y_pred: Annotated[int, Field(description="Model prediction")]
    rationale_true: Annotated[str, Field(description="Human rationale annotation")]
    rationale_pred: Annotated[
        str, Field(description="Model rationale for the prediction")
    ]


class DemoData(Immutable):
    """Entry for the data used by the demonstration tool."""

    graph: gpt.Graph
    related: Sequence[gpt.PaperRelatedSummarised]
    paper: DemoPaper


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Graph evaluation output.")
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSON file."),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n", help="How many items to sample. Defaults to all."
        ),
    ] = 10,
    seed: Annotated[int, typer.Option(help="Random seed used for sampling")] = 0,
) -> None:
    """Extract only relevant data from the input file."""
    rng = random.Random(seed)

    papers = sample(
        gpt.PromptResult.unwrap(
            load_data(input_file, gpt.PromptResult[gpt.GraphResult])
        ),
        limit,
        rng,
    )
    papers_valid = [
        p
        for p in papers
        if p.paper.y_pred == p.paper.y_true
        and is_rationale_valid(p.rationale_pred)
        and p.related
    ]
    papers_converted = [
        DemoData(
            graph=p.graph,
            related=p.related or [],
            paper=DemoPaper(
                title=p.paper.title,
                abstract=p.paper.abstract,
                authors=p.paper.authors,
                sections=p.paper.sections,
                approval=p.paper.approval,
                conference=_clean_conference(p.paper.conference),
                rating=p.paper.rating,
                year=_get_year(p.paper),
                id=p.id,
                y_true=p.paper.y_true,
                y_pred=p.paper.y_pred,
                rationale_true=p.paper.rationale_true,
                rationale_pred=p.rationale_pred,
            ),
        )
        for p in papers_valid
    ]
    save_data(output_file, papers_converted, compress=Compress.AUTO)


if __name__ == "__main__":
    app()
