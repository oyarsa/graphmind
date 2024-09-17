"""Get ACL Anthology paper information from Semantic Scholar API.

Does not include the full text of the papers.
"""

import argparse

from semanticscholar import SemanticScholar

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


def search_papers(query: str, conferences: list[str], year: str, n: int) -> None:
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
    for i, item in enumerate(items[:n]):
        print(
            f"{i+1}. {item.title}\nYear: {item.year}\nVenue: {item.venue}"
            f"\nPDF: {item.openAccessPdf["url"]}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", default="*", help="Query string to search for")
    parser.add_argument(
        "--conferences",
        nargs="+",
        default=_ACL_CONFERENCES,
        help="List of ACL conferences to search for",
    )
    parser.add_argument("--year", default="", help="Year to search for")
    parser.add_argument("--n", type=int, default=10, help="Number of papers to display")

    args = parser.parse_args()
    search_papers(args.query, args.conferences, args.year, args.n)


if __name__ == "__main__":
    main()
