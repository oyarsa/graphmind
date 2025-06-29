"""Download LaTeX source files from arXiv."""

import dataclasses as dc
import io
import itertools
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import arxiv  # type: ignore
import backoff
import orjson
import requests
import typer
from tqdm import tqdm

from paper.util.serde import read_file_bytes

logger = logging.getLogger(__name__)


@dc.dataclass(frozen=True, kw_only=True)
class ArxivResult:
    """Result of querying the arXiv API with a paper title from OpenReview."""

    openreview_title: str
    arxiv_title: str
    id: str
    summary: str | None = None


def latex(
    reviews_file: Annotated[
        Path,
        typer.Option(
            "--arxiv-ids",
            "-i",
            help="Path to data from arXiv used to download the LaTeX files.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Output directory for arXiv LaTeX source files."
        ),
    ],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Download LaTeX source files from arXiv data.

    The arXiv data is fetched with the `arxiv` subcommand.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean`.
    """
    papers: list[dict[str, str]] = orjson.loads(read_file_bytes(reviews_file))[
        :max_papers
    ]
    arxiv_results = [
        ArxivResult(
            openreview_title=p["openreview_title"],
            arxiv_title=p["arxiv_title"],
            id=p["arxiv_id"],
        )
        for p in papers
    ]

    if clean_run:
        downloaded_prev: set[str] = set()
    else:
        downloaded_prev = {
            path.stem for path in output_dir.glob("*.tar.gz") if path.is_file()
        }

    downloaded_n = 0
    skipped_n = 0
    failed_n = 0
    output_dir.mkdir(exist_ok=True, parents=True)

    for result in tqdm(arxiv_results, desc="Downloading LaTeX sources"):
        if result.arxiv_title in downloaded_prev:
            skipped_n += 1
            continue

        try:
            if data := download_latex_source(result.id):
                (output_dir / f"{result.arxiv_title}.tar.gz").write_bytes(data)
                downloaded_n += 1
            else:
                logger.warning(f"Invalid tar.gz file for {result.arxiv_title}")
                failed_n += 1
        except Exception as e:
            logger.warning(
                f"Error downloading LaTeX source for {result.arxiv_title}"
                f" - {type(e).__name__}: {e}"
            )
            failed_n += 1

    logger.info(f"Downloaded : {downloaded_n}")
    logger.info(f"Skipped    : {skipped_n}")
    logger.info(f"Failed     : {failed_n}")


def latex_all(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to directory with data downloaded with `reviews-all`.",
        ),
    ],
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="How many papers to process. If None, processes all.",
        ),
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option("--clean", help="If True, ignore previously downloaded files."),
    ] = False,
) -> None:
    """Download LaTeX source files from arXiv for data downloaded from `reviews-all`.

    The `--input` parameter should be the same directory as the `output_dir` from
    `reviews-all`.

    By default, skips re-downloading files that already exist in the output directory.
    You can override this with `--clean`.
    """
    venue_dirs = list(data_dir.iterdir())
    for i, venue_dir in enumerate(venue_dirs, 1):
        logger.info("\n>>> [%d/%d] %s", i, len(venue_dirs), venue_dir.name)

        arxiv_file = venue_dir / "openreview_arxiv.json"
        if not arxiv_file.exists():
            logger.warning("No arXiv data file for: %s", venue_dir)
            continue

        latex(arxiv_file, venue_dir / "files", max_papers, clean_run)


def get_arxiv(openreview_titles: list[str], batch_size: int) -> dict[str, ArxivResult]:
    """Get mapping of OpenReview paper title (casefold) to arXiv ID that exist on arXiv."""
    arxiv_client = arxiv.Client()

    arxiv_results: list[ArxivResult] = []
    for openreview_title_batch in tqdm(
        list(itertools.batched(openreview_titles, batch_size)),
        desc="Querying arXiv",
    ):
        arxiv_results.extend(
            _batch_search_arxiv(arxiv_client, list(openreview_title_batch))
        )

    return {normalise_title(r.openreview_title): r for r in arxiv_results}


@backoff.on_exception(backoff.expo, Exception, max_tries=5, logger=logger)
def arxiv_search(
    client: arxiv.Client, query: str, max_results: int
) -> Iterator[arxiv.Result]:
    """Query up to `max_results` matches of `query` on arXiv with retries."""
    logger.info("Querying n=%d query='%s'", max_results, query)
    return client.results(arxiv.Search(query=f'ti:"{query}"', max_results=max_results))


@backoff.on_exception(backoff.expo, Exception, max_tries=5, logger=logger)
def arxiv_from_id(client: arxiv.Client, arxiv_id: str) -> Iterator[arxiv.Result]:
    """Query paper ID on arXiv with retries."""
    return client.results(arxiv.Search(id_list=[arxiv_id]))


def _batch_search_arxiv(
    client: arxiv.Client, openreview_titles: list[str]
) -> list[ArxivResult]:
    """Search multiple OpenReview titles at once on arXiv and return matching results."""
    or_queries = " OR ".join(
        f'ti:"{openreview_title}"' for openreview_title in openreview_titles
    )
    query = f"({or_queries})"
    results_map: dict[str, ArxivResult] = {}
    openreview_titles_norm = [normalise_title(t) for t in openreview_titles]

    try:
        for result in arxiv_search(client, query, len(openreview_titles)):
            arxiv_title = result.title
            for i, openreview_title in enumerate(openreview_titles_norm):
                if similar_titles(openreview_title, arxiv_title):
                    results_map[openreview_title] = ArxivResult(
                        id=arxiv_id_from_url(result.entry_id),
                        openreview_title=openreview_titles[i],
                        arxiv_title=arxiv_title,
                        summary=result.summary,
                    )
                    break
    except Exception as e:
        logger.warning(f"Error during batch search on arXiv: {e}")

    return [
        result
        for openreview_title in openreview_titles_norm
        if (result := results_map.get(openreview_title))
    ]


def arxiv_id_from_url(url: str) -> str:
    """Parse arXiv paper ID from the URL (abstract, PDF, etc.)."""
    return url.split("/")[-1]


def similar_titles(title1: str, title2: str) -> bool:
    """Check if two titles are similar (case-insensitive, stripped)."""
    t1 = title1.casefold().strip()
    t2 = title2.casefold().strip()
    return t1 == t2 or t1 in t2 or t2 in t1


@backoff.on_exception(backoff.expo, Exception, max_tries=5, logger=logger)
def download_latex_source(arxiv_id: str) -> bytes | None:
    """Download LaTeX source (tar.gz) from arXiv for the given arXiv ID."""
    url = f"https://arxiv.org/src/{arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    content = response.content

    if not _is_valid_targz(content):
        return None
    return content


def _is_valid_targz(content: bytes) -> bool:
    """Check if the given content is a valid tar.gz file.

    Args:
        content: Binary content (bytes) to check.

    Returns:
        True if the content is a valid tar.gz file, False otherwise.
    """
    try:
        file_like_object = io.BytesIO(content)
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            # If we can list the contents, it's a valid tar.gz file
            tar.getnames()
    except (tarfile.ReadError, tarfile.CompressionError, EOFError):
        return False
    else:
        return True


def normalise_title(title: str) -> str:
    """Normalize title for comparison."""
    return title.casefold()
