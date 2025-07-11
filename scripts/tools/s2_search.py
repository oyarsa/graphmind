"""Test script for Semantic Scholar paper search using fetch_paper_data."""

import asyncio
import json
import logging
import sys
from typing import Annotated, Any

import aiohttp
import dotenv
import typer

from paper.semantic_scholar.info import fetch_paper_data
from paper.util import ensure_envvar, setup_logging

# Fields from paper_retrieval.py
S2_FIELDS = [
    "paperId",
    "corpusId",
    "url",
    "title",
    "authors",
    "year",
    "abstract",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "tldr",
    "venue",
]

REQUEST_TIMEOUT = 60


async def search_papers(
    title: str,
    limit: int = 1,
) -> None:
    """Search for papers by title and print results as JSON."""
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT)
    ) as session:
        limiter = asyncio.Semaphore(10)  # Rate limiter

        print(f"Searching for: '{title}' (limit: {limit})", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        results: list[dict[str, Any]] = []

        # Fetch multiple results if limit > 1
        data = await fetch_paper_data(
            session=session,
            api_key=api_key,
            paper_title=title,
            fields=S2_FIELDS,
            limiter=limiter,
            limit=limit,
        )

        if data:
            # The function returns a single result with title_query added
            # For multiple results, we need to modify the approach
            # Let's make a direct API call to get multiple results
            params = {
                "query": title,
                "fields": ",".join(S2_FIELDS),
                "limit": limit,
            }
            headers = {"x-api-key": api_key}

            async with (
                limiter,
                session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params,
                    headers=headers,
                ) as response,
            ):
                if response.status == 200:
                    full_data = await response.json()
                    if full_data.get("data"):
                        for paper in full_data["data"]:
                            paper["title_query"] = title
                            results.append(paper)

        if results:
            # Pretty print JSON to stdout
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("[]")  # Empty array for no results
            print(f"No results found for '{title}'", file=sys.stderr)


def main(
    title: Annotated[str, typer.Argument(help="Paper title to search for")],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of results to return",
            min=1,
            max=100,
        ),
    ] = 5,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Search Semantic Scholar for papers by title and output JSON results."""
    dotenv.load_dotenv()

    if verbose:
        setup_logging()
        logging.getLogger("paper.semantic_scholar").setLevel(logging.DEBUG)
    else:
        # Suppress all logging to stderr except critical errors
        logging.basicConfig(level=logging.CRITICAL)

    try:
        asyncio.run(search_papers(title, limit))
    except KeyboardInterrupt:
        print("\nSearch cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
