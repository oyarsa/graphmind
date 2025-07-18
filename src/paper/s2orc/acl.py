"""Get ACL Anthology paper information from Semantic Scholar API.

Does not include the full text of the papers.
"""

# Because SemanticScholar is completely untyped, and it's not worth wrapping it
# pyright: basic

from typing import Annotated

import typer
from semanticscholar import SemanticScholar
from semanticscholar.PaginatedResults import PaginatedResults

from paper.util.cli import die

# From https://aclanthology.org/
_ACL_CONFERENCES = [
    "ACL",
    "AACL",
    "ANLP",
    "CL",
    "CoNLL",
    "EACL",
    "EMNLP",
    "Findings",
    "IWSLT",
    "NAACL",
    "SemEval",
    "*SEM",
    "TACL",
    "WMT",
    "WS",
    "ALTA",
    "AMTA",
    "CCL",
    "COLING",
    "EAMT",
    "HLT",
    "IJCLCLP",
    "IJCNLP",
    "JEP/TALN/RECITAL",
    "KONVENS",
    "LILT",
    "LREC",
    "MTSummit",
    "MUC",
    "NEJLT",
    "PACLIC",
    "RANLP",
    "ROCLING",
    "TAL",
    "TINLAP",
    "TIPSTER",
]


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    query: Annotated[str, typer.Option(help="Query string to search for")] = "*",
    conferences: Annotated[
        list[str], typer.Option(help="List of ACL conferences to search for")
    ] = _ACL_CONFERENCES,
    year: Annotated[str, typer.Option(help="Year to search for")] = "",
    limit: Annotated[
        int, typer.Option("--limit", "-n", help="Number of papers to display")
    ] = 10,
) -> None:
    """Search papers in the Semantic Scholar database from a text query.

    Args:
        query: Query string to search for.
        conferences: List of conferences to match. This should not be empty.
        year: Year to match paper publication date.
        limit: Maximum number of papers to display.

    Raises:
        SystemExit: if `conferences` is empty.
    """
    if not conferences:
        die("Conferences should not be empty.")

    sch = SemanticScholar()

    results = sch.search_paper(
        query,
        venue=conferences,
        year=year,
        bulk=True,
        sort="citationCount:desc",
    )
    assert isinstance(results, PaginatedResults)

    print("Top-10 by citationCount:\n")

    items = [p for p in results.items if p.isOpenAccess and p.openAccessPdf is not None]
    for i, item in enumerate(items[:limit]):
        print(
            f"{i + 1}. {item.title}\nYear: {item.year}\nVenue: {item.venue}"
            f"\nPDF: {item.openAccessPdf['url']}\n"
        )


if __name__ == "__main__":
    app()
