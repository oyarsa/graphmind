"""Download paper information from the Semantic Scholar (S2) API.

The input is the output of the ASAP pipeline[1], asap_filtered.json. The goal is to get
extended information from the API (e.g. the paperId) so we can use in future queries
(e.g. get recommendations from the paperId).

We take the titles of all the ASAP papers, then query the S2 search API with them. S2
does their own paper title matching, so we just take the best match directly. We then
save the entire output to a JSON file along with the original data from ASAP. The data
has the same shape as the input ASAP Paper, with the S2 data under a new `s2` key.

The resulting files (under `output_dir`) are:
- full.json: all the input data, optionally with the S2 data if it could be found.
- valid.json: only entries with valid (non-empty) results with fuzzy ratios[2].
- final.json: the valid results with a non-empty abstract and fuzzy ratio[2] above a
  minimum (default: 80).

[1] See `paper.asap.preprocess`.
[2] See `paper.util.title_ratio`.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiohttp
import dotenv
from pydantic import ConfigDict, TypeAdapter

from paper.asap.model import Paper as ASAPPaper
from paper.external_data.semantic_scholar_model import Paper as S2Paper
from paper.external_data.semantic_scholar_model import title_ratio
from paper.progress import gather
from paper.util import (
    HelpOnErrorArgumentParser,
    arun_safe,
    display_params,
    ensure_envvar,
    get_limiter,
    setup_logging,
)

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60  # 1 minute timeout for each request
MAX_RETRIES = 5
RETRY_DELAY = 5  # Initial delay in seconds
BACKOFF_FACTOR = 2  # Exponential backoff factor

S2_SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

LIMITER = get_limiter(MAX_CONCURRENT_REQUESTS)
logger = logging.getLogger(__name__)


class ASAPPaperWithS2(ASAPPaper):
    model_config = ConfigDict(frozen=True)

    s2: S2Paper | None
    fuzz_ratio: int


async def _fetch_paper_info(
    session: aiohttp.ClientSession,
    api_key: str,
    paper_title: str,
    fields: Sequence[str],
) -> dict[str, Any] | None:
    """Fetch paper information for a given title. Takes only the best title match."""
    params = {
        "query": paper_title,
        "fields": ",".join(fields),
        "limit": 1,  # We're only interested in the best match
    }
    headers = {"x-api-key": api_key}

    async with LIMITER:
        attempt = 0
        delay = RETRY_DELAY
        while attempt < MAX_RETRIES:
            try:
                async with session.get(
                    S2_SEARCH_BASE_URL, params=params, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data"):
                            return {"title_query": paper_title} | data["data"][0]
                        else:
                            logger.debug(f"No results found for title: {paper_title}")
                            return None
                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                            wait_source = "Retry-After"
                        else:
                            wait_time = delay
                            wait_source = "Custom"
                        logger.debug(
                            f"Rate limited (429) when fetching '{paper_title}'. "
                            f"Retrying after {wait_time} ({wait_source}) seconds..."
                        )
                        await asyncio.sleep(wait_time)
                        delay *= BACKOFF_FACTOR  # Exponential backoff
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"Error fetching data for '{paper_title}': HTTP {response.status} - {error_text}"
                        )
                        return None
            except aiohttp.ClientError as e:
                logger.debug(
                    f"Network error fetching '{paper_title}': {e}. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR
            except TimeoutError:
                logger.debug(
                    f"Timeout error fetching '{paper_title}'. Retrying..."
                    f" (Attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR

            attempt += 1

        logger.warning(
            f"Failed to fetch data for '{paper_title}' after {MAX_RETRIES} attempts."
        )
        return None


async def _download_paper_info(
    input_file: Path,
    fields_str: str,
    output_path: Path,
    api_key: str,
    min_fuzzy: int,
    limit_papers: int | None,
) -> None:
    """Download paper information for multiple titles.

    The input file is the output of the ASAP preprocessing pipeline (asap_filtered.json).
    We obtain the unique titles from the references of all papers and get their info.

    The API allows us to specify the returned fields, so pass just the relevant ones
    to minimise the bandwidth required.
    """
    logger.info(display_params())

    fields = [f for field in fields_str.split(",") if (f := field.strip())]
    papers_asap = TypeAdapter(list[ASAPPaper]).validate_json(input_file.read_bytes())
    papers_asap = papers_asap[:limit_papers]

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(REQUEST_TIMEOUT),
        connector=aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS),
    ) as session:
        tasks = [
            _fetch_paper_info(session, api_key, paper.title, fields)
            for paper in papers_asap
        ]
        task_results = list(await gather(tasks, desc="Downloading paper info"))
        results = [
            ASAPPaperWithS2(
                title=paper.title,
                abstract=paper.abstract,
                reviews=paper.reviews,
                sections=paper.sections,
                approval=paper.approval,
                references=paper.references,
                s2=S2Paper.model_validate(result) if result else None,
                fuzz_ratio=title_ratio(paper.title, result["title"]) if result else 0,
            )
            for paper, result in zip(papers_asap, task_results)
        ]

    results_valid = [result for result in results if result.s2]
    results_filtered = [
        paper
        for paper in results_valid
        if paper.fuzz_ratio >= min_fuzzy and paper.s2 and paper.s2.abstract
    ]

    logger.info(f"{len(results)} papers")
    logger.info(
        f"{len(results_valid)} valid (non-null results)"
        f" - {len(results_valid)/len(results):.2%} papers found something"
    )
    logger.info(
        f"{len(results_filtered)} filtered (non-empty abstract and fuzz ratio"
        f" >= {min_fuzzy})"
    )
    logger.info(f"> {len(results_filtered)/len(results):.2%} papers filtered")

    output_path.mkdir(parents=True, exist_ok=True)
    _save_result(output_path, "full.json", results)
    _save_result(output_path, "valid.json", results_valid)
    _save_result(output_path, "filtered.json", results_filtered)


def _save_result(path: Path, name: str, data: list[ASAPPaperWithS2]) -> None:
    (path / name).write_bytes(
        TypeAdapter(list[ASAPPaperWithS2]).dump_json(data, indent=2)
    )


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument("input_file", type=Path, help="Input file (asap_filtered.json)")
    parser.add_argument(
        "output_path", type=Path, help="Directory to save the downloaded information"
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=",".join(
            (
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
            )
        ),
        help="Comma-separated list of fields to retrieve",
    )
    parser.add_argument(
        "--min-fuzzy",
        type=int,
        default=80,
        help="Minimum fuzz ratio of titles to filter",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of papers to query",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")
    setup_logging()

    arun_safe(
        _download_paper_info,
        args.input_file,
        args.fields,
        args.output_path,
        api_key,
        args.min_fuzzy,
        args.limit,
    )


if __name__ == "__main__":
    main()
