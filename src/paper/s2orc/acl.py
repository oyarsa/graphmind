"""Get ACL Anthology paper information from Semantic Scholar API.

Does not include the full text of the papers.
"""

# Because SemanticScholar is completely untyped, and it's not worth wrapping it
# pyright: basic

from semanticscholar import SemanticScholar

from paper.util import HelpOnErrorArgumentParser

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


def search_papers(query: str, conferences: list[str], year: str, limit: int) -> None:
    """Search papers in the Semantic Scholar database from a text query.

    Args:
        query: Query string to search for.
        conferences: List of conferences to match.
        year: Year to match paper publication date.
        limit: Maximum number of papers to display.
    """
    sch = SemanticScholar()

    results = sch.search_paper(
        query,
        venue=conferences,
        year=year,
        bulk=True,
        sort="citationCount:desc",
    )

    print("Top-10 by citationCount:\n")

    items = [p for p in results.items if p.isOpenAccess and p.openAccessPdf is not None]
    for i, item in enumerate(items[:limit]):
        print(
            f"{i+1}. {item.title}\nYear: {item.year}\nVenue: {item.venue}"
            f"\nPDF: {item.openAccessPdf["url"]}\n"
        )


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument("--query", default="*", help="Query string to search for")
    parser.add_argument(
        "--conferences",
        nargs="+",
        default=_ACL_CONFERENCES,
        help="List of ACL conferences to search for",
    )
    parser.add_argument("--year", default="", help="Year to search for")
    parser.add_argument(
        "--limit", "-n", type=int, default=10, help="Number of papers to display"
    )

    args = parser.parse_args()
    search_papers(args.query, args.conferences, args.year, args.limit)


if __name__ == "__main__":
    main()
