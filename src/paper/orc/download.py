"""Download paper data from OpenReview."""
# pyright: basic

import dataclasses as dc
import itertools
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Any

import openreview as openreview_v1  # type: ignore
import orjson
import typer
from openreview import api as openreview_v2  # type: ignore
from tqdm import tqdm

from paper.orc.arxiv_api import (
    ArxivResult,
    download_latex_source,
    get_arxiv,
    normalise_title,
)
from paper.orc.latex_parser import (
    SentenceSplitter,
    latex_paper_to_peerread,
    process_latex,
)
from paper.peerread.model import Paper, PaperReference, PaperSection
from paper.semantic_scholar.info import (
    fetch_arxiv_papers,
)
from paper.util import arun_safe, ensure_envvar
from paper.util.serde import write_file_bytes

logger = logging.getLogger(__name__)

# Valid keys for target ratings
RATING_KEYS = [
    "contribution",
    "technical_novelty_and_significance",
    "empirical_novelty_and_significance",
]
"""Valid keys for target ratings.

ICLR 2024 and 2025, and NeurIPS 2022-2024 use 'contribution'.
ICLR 2022 and 2023 use technical/empirical novelty.

Our target rating is the max of the available ratings.
"""


def reviews(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    venue_id: Annotated[str, typer.Option("--venue", help="Venue ID to fetch data.")],
    batch_size: Annotated[
        int, typer.Option(help="Batch size to query the arXiv API.")
    ] = 50,
    max_papers: Annotated[
        int | None,
        typer.Option(
            "--max-papers",
            "-n",
            help="Maximum number of papers to query. If None, use all.",
        ),
    ] = None,
    query_arxiv: Annotated[
        bool, typer.Option("--arxiv/--no-arxiv", help="Query arXiv for papers")
    ] = True,
) -> None:
    """Download all reviews and metadata for papers from a conference in OpenReview.

    Also queries arXiv to find which papers are available there, adding their arXiv IDs.

    Requires the following environment variables to be set:
    - OPENREVIEW_USERNAME
    - OPENREVIEW_PASSWORD

    These are the standard credentials you use to log into the OpenReview website.

    Supports conference using both v1 and v2 APIs.
    Example venue IDs:

    API v1:
    - ICLR.cc/2022/Conference
    - ICLR.cc/2023/Conference
    - NeurIPS.cc/2022/Conference
    - NeurIPS.cc/2023/Conference

    API v2:
    - ICLR.cc/2024/Conference
    - ICLR.cc/2025/Conference
    - NeurIPS.cc/2024/Conference
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    submissions_raw = get_conference_submissions(venue_id)
    if not submissions_raw:
        logger.warning("No submissions available for %s", venue_id)
        return

    submissions_all = [_note_to_dict(s) for s in submissions_raw]
    logger.info("Submissions - all: %d", len(submissions_all))
    write_file_bytes(
        output_dir / "openreview_all.json.zst", orjson.dumps(submissions_all)
    )

    submissions_valid = [s for s in submissions_all if is_valid(s, RATING_KEYS)]
    logger.info("Submissions - valid: %d", len(submissions_valid))
    write_file_bytes(
        output_dir / "openreview_valid.json.zst",
        orjson.dumps(submissions_valid),
    )

    if not submissions_valid:
        logger.warning("No valid submissions for %s", venue_id)
        return

    openreview_titles = [
        openreview_title
        for paper in submissions_valid[:max_papers]
        if (openreview_title := get_value(paper["content"], "title"))
    ]

    if not query_arxiv:
        logger.info("Skipping arXiv query.")
        return

    logger.info("Querying arXiv for %d paper titles", len(openreview_titles))
    openreview_to_arxiv = get_arxiv(openreview_titles, batch_size)
    logger.info("Found %d papers on arXiv", len(openreview_to_arxiv))
    write_file_bytes(
        output_dir / "openreview_arxiv.json.zst",
        orjson.dumps([dc.asdict(v) for v in openreview_to_arxiv.values()]),
    )

    submissions_with_arxiv: list[dict[str, Any]] = []
    for paper in submissions_valid:
        openreview_title = get_value(paper["content"], "title") or ""
        if arxiv_result := openreview_to_arxiv.get(normalise_title(openreview_title)):
            submissions_with_arxiv.append({
                **paper,
                "arxiv_id": arxiv_result.id,
                "openreview_title": arxiv_result.openreview_title,
                "arxiv_title": arxiv_result.arxiv_title,
            })

    logger.info("Submissions with arXiv IDs: %d", len(submissions_with_arxiv))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "openreview_arxiv.json").write_bytes(
        orjson.dumps(submissions_with_arxiv)
    )


def reviews_all(
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for OpenReview reviews file.")
    ],
    query_arxiv: Annotated[
        bool, typer.Option("--arxiv/--no-arxiv", help="Query arXiv for papers")
    ] = True,
) -> None:
    """Download reviews and arXiv IDs for the following conferences.

    - ICLR 2022, 2023, 2024, 2025
    - NeurIPS 2022, 2023, 2024

    Each one gets its own subdirectory under `output_dir`.
    """
    # <venue ID, directory name>
    conferences = [
        ("ICLR.cc/2022", "iclr2022"),
        ("ICLR.cc/2023", "iclr2023"),
        ("ICLR.cc/2024", "iclr2024"),
        ("ICLR.cc/2025", "iclr2025"),
        ("NeurIPS.cc/2022", "neurips2022"),
        ("NeurIPS.cc/2023", "neurips2023"),
        ("NeurIPS.cc/2024", "neurips2024"),
    ]

    with tqdm(total=len(conferences)) as pbar:
        for venue_id, dir_name in conferences:
            pbar.set_description(f"{venue_id}")

            dir_path = output_dir / dir_name
            if dir_path.exists():
                pbar.update(1)
                continue

            reviews(dir_path, f"{venue_id}/Conference", query_arxiv=query_arxiv)
            pbar.update(1)


def get_conference_submissions(
    venue_id: str,
) -> list[openreview_v1.Note | openreview_v2.Note]:
    """Get all submissions with reviews for `venue_id`.

    Tries both API versions and submissions/blind submissions.
    """
    clients = [
        openreview_v2.OpenReviewClient(baseurl="https://api2.openreview.net"),
        openreview_v1.Client(baseurl="https://api.openreview.net"),
    ]
    sections = ["Submission", "Blind_Submission"]
    details = ["replies", "directReplies"]

    submissions_all: list[openreview_v1.Note | openreview_v2.Note] = []

    for client, section, detail in itertools.product(clients, sections, details):
        if submissions := client.get_all_notes(  # type: ignore
            invitation=f"{venue_id}/-/{section}", details=detail
        ):
            submissions_all.extend(submissions)  # type: ignore

    # Querying both replies and directReplies might yield duplicate papers, so let's
    # keep only the first seen.
    title_to_paper: dict[str, openreview_v1.Note | openreview_v2.Note] = {}

    for paper in submissions_all:
        title: str | None = get_value(paper.content, "title")  # type: ignore
        if title and title not in title_to_paper:
            title_to_paper[title] = paper

    return list(title_to_paper.values())


def _note_to_dict(note: openreview_v1.Note | openreview_v2.Note) -> dict[str, Any]:
    """Convert OpenReview API `Note` to dict with additional `details` object."""
    return note.to_json() | {"details": note.details}


def is_valid(paper: dict[str, Any], rating_keys: list[str]) -> bool:
    """Check if paper has at least one review with `rating`, PDF, title and abstract."""
    return all((
        any(_has_rating(paper, key) for key in rating_keys),
        _has_rating(paper, "decision"),
        _has_field(paper, "pdf"),
        _has_field(paper, "title"),
        _has_field(paper, "abstract"),
    ))


def get_reviews(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Get reviews from paper object: either `replies` or `directReplies`."""
    details = paper["details"]
    if "replies" in details:
        return paper["details"]["replies"]
    if "directReplies" in details:
        return paper["details"]["directReplies"]

    raise ValueError("Paper does not have review replies")


def _has_rating(paper: dict[str, Any], name: str) -> bool:
    """Check if any review in `paper` has the rating with given `name`."""
    return any(r["content"].get(name) for r in get_reviews(paper))


def get_value(item: dict[str, Any], key: str) -> Any | None:
    """Get value from OpenReview API response.

    If the item is a dict with a 'value' key, returns that value.
    Otherwise returns the item directly.
    """
    value = item.get(key, {})
    if isinstance(value, dict):
        return value.get("value")
    return value


def _has_field(paper: dict[str, Any], name: str) -> bool:
    """Check if the `paper` has a field with `name` and non-empty value."""
    value = get_value(paper["content"], name)
    if isinstance(value, str):
        value = value.strip()
    return bool(value)


def parse_arxiv_latex(
    arxiv_result: ArxivResult, splitter: SentenceSplitter
) -> tuple[list[PaperSection], list[PaperReference]]:
    """Download and parse arXiv LaTeX for a paper, returning sections and references.

    Args:
        arxiv_result: arXiv paper information.
        splitter: Sentence splitter for parsing.

    Returns:
        Tuple of (sections, references) from parsed LaTeX.

    Raises:
        RuntimeError: If LaTeX download or parsing fails.
    """
    # Download the LaTeX source as bytes
    latex_bytes = download_latex_source(arxiv_result.id)
    if not latex_bytes:
        raise RuntimeError(f"Failed to download LaTeX for arXiv ID: {arxiv_result.id}")

    # Create temporary file to store the tarball
    with tempfile.TemporaryDirectory(suffix=".tar.gz") as tmp_dir:
        tmp_file = Path(tmp_dir) / "latex.tar.gz"
        tmp_file.write_bytes(latex_bytes)

        # Parse the LaTeX using the existing parser
        latex_paper = process_latex(splitter, arxiv_result.arxiv_title, tmp_file)
        if not latex_paper:
            raise RuntimeError(
                f"Failed to parse LaTeX for paper: {arxiv_result.arxiv_title}"
            )

        # Convert to PeerRead format using the helper function
        return latex_paper_to_peerread(latex_paper)


async def _download_papers_from_titles(
    titles: list[str],
    output_dir: Path,
    batch_size: int = 50,
) -> None:
    """Download paper metadata from Semantic Scholar and parse arXiv LaTeX for titles."""
    api_key = ensure_envvar("SEMANTIC_SCHOLAR_API_KEY")

    # Fields to retrieve from S2 API
    fields = [
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

    logger.info("Fetching data from Semantic Scholar for %d titles", len(titles))

    s2_results = await fetch_arxiv_papers(
        api_key, titles, fields, desc="Downloading from S2"
    )
    s2_results_valid = [p for p in s2_results if p]

    logger.info("Found %d papers on Semantic Scholar", len(s2_results_valid))

    s2_titles = [paper.title for paper in s2_results_valid]
    logger.info("Querying arXiv for %d paper titles", len(s2_titles))

    openreview_to_arxiv = get_arxiv(s2_titles, batch_size)
    logger.info("Found %d papers on arXiv", len(openreview_to_arxiv))

    if not openreview_to_arxiv:
        logger.warning(
            "No papers found on arXiv. Cannot proceed without LaTeX content."
        )
        return

    papers: list[Paper] = []
    splitter = SentenceSplitter()

    for s2_paper in s2_results_valid:
        # Only process papers that have arXiv matches
        normalized_title = normalise_title(s2_paper.title)
        arxiv_result = openreview_to_arxiv.get(normalized_title)

        if not arxiv_result:
            logger.debug("No arXiv match for S2 paper: %s", s2_paper.title)
            continue

        logger.info("Processing paper: %s (arXiv: %s)", s2_paper.title, arxiv_result.id)

        try:
            sections, references = parse_arxiv_latex(arxiv_result, splitter)
            papers.append(
                Paper.from_s2(
                    s2_paper,
                    sections=sections,
                    references=references,
                )
            )
        except RuntimeError as e:
            logger.warning("Failed to process paper %s: %s", s2_paper.title, e)
            continue

    write_file_bytes(
        output_dir / "papers_from_titles.json.zst",
        orjson.dumps([paper.model_dump() for paper in papers]),
    )
    logger.info("Saved %d papers to %s", len(papers), output_dir)


def reviews_from_titles(
    titles_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="File containing paper titles (one per line).",
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for paper data.")
    ],
    batch_size: Annotated[
        int, typer.Option(help="Batch size to query the arXiv API.")
    ] = 50,
) -> None:
    """Download paper metadata from Semantic Scholar and parse arXiv LaTeX given a list of titles.

    Input file should contain one paper title per line.
    Creates Paper objects with S2 metadata and parsed arXiv sections/references (no reviews).
    Only processes papers that have matches on both Semantic Scholar and arXiv.

    Requires SEMANTIC_SCHOLAR_API_KEY environment variable.
    """
    titles = [
        title
        for line in titles_file.read_text(encoding="utf-8").splitlines()
        if (title := line.strip())
    ]

    if not titles:
        logger.warning("No titles found in %s", titles_file)
        return

    logger.info("Querying %d titles from %s", len(titles), titles_file)

    arun_safe(_download_papers_from_titles, titles, output_dir, batch_size)
