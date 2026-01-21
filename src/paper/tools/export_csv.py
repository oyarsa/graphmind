"""Export GPT graph evaluation results to CSV for annotation pipelines.

Creates a CSV with one row per paper containing:
- Paper metadata (title, authors, abstract, PDF link)
- Ground truth review and rating
- Model predicted rationale and rating
- Graph components (claims, methods, experiments) as bullet points
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

import arxiv  # type: ignore
import typer
from tqdm import tqdm

from paper.gpt.model import PromptResult
from paper.util import setup_logging
from paper.util.serde import load_data

logger = logging.getLogger("paper.export_csv")

ARXIV_CACHE_FILE = Path("output/arxiv_title_cache.json")


def _normalise_title(title: str) -> str:
    """Normalise title for comparison and caching."""
    return title.casefold().strip()


def _load_arxiv_cache(cache_file: Path) -> dict[str, str]:
    """Load arxiv ID cache from JSON file."""
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_arxiv_cache(cache: dict[str, str], cache_file: Path) -> None:
    """Save arxiv ID cache to JSON file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _search_arxiv_single(client: arxiv.Client, title: str) -> str | None:
    """Search arXiv for a single title and return arxiv ID if found."""
    try:
        search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
        for result in client.results(search):
            arxiv_title = result.title.casefold().strip()
            query_title = title.casefold().strip()
            if (
                arxiv_title == query_title
                or arxiv_title in query_title
                or query_title in arxiv_title
            ):
                return result.entry_id.split("/abs/")[-1]
    except Exception as e:
        logger.warning(f"Error searching arXiv for '{title[:50]}...': {e}")
    return None


def _lookup_arxiv_ids(
    titles: list[str], cache_file: Path, rate_limit: float = 3.1
) -> dict[str, str]:
    """Look up arxiv IDs for titles, using cache and querying API for missing ones."""
    cache = _load_arxiv_cache(cache_file)
    result: dict[str, str] = {}
    missing_titles: list[str] = []

    for title in titles:
        norm_title = _normalise_title(title)
        if norm_title in cache:
            result[norm_title] = cache[norm_title]
        else:
            missing_titles.append(title)

    if missing_titles:
        logger.info(f"Looking up {len(missing_titles)} titles on arXiv...")
        client = arxiv.Client()
        found = 0

        for title in tqdm(missing_titles, desc="Querying arXiv"):
            norm_title = _normalise_title(title)
            arxiv_id = _search_arxiv_single(client, title)

            if arxiv_id:
                result[norm_title] = arxiv_id
                cache[norm_title] = arxiv_id
                found += 1
            else:
                result[norm_title] = ""
                cache[norm_title] = ""

            time.sleep(rate_limit)

        _save_arxiv_cache(cache, cache_file)
        logger.info(f"Found {found}/{len(missing_titles)} arxiv IDs, cached results")

    return result


def _format_entities_as_bullets(
    entities: list[dict[str, Any]], entity_type: str
) -> str:
    """Extract entities of given type and format as bullet points (labels only)."""
    labels = [e["label"] for e in entities if e["type"] == entity_type]
    if not labels:
        return ""
    return "\n".join(f"• {label}" for label in labels)


def _format_authors(authors: list[str]) -> str:
    """Format author list for display."""
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]}, {authors[1]}, ... ({len(authors)} authors)"


def _has_valid_graph(graph: dict[str, Any]) -> bool:
    """Check if graph has at least one entity."""
    entities = graph.get("entities", [])
    return len(entities) > 0


def _has_valid_rationales(paper: dict[str, Any]) -> bool:
    """Check if paper has non-empty ground truth and predicted rationales."""
    ground_truth = paper.get("rationale", "")
    predicted = paper.get("rationale_pred", "")
    return bool(
        ground_truth and ground_truth.strip() and predicted and predicted.strip()
    )


def _truncate_evidence_sections(rationale: str, max_papers: int = 5) -> str:
    """Truncate supporting/contradictory evidence sections to max_papers each.

    Looks for bullet point lists (lines starting with -) within evidence sections
    and keeps only the first max_papers items in each section.
    """
    import re

    lines = rationale.split("\n")
    result_lines: list[str] = []
    in_evidence_section = False
    bullet_count = 0

    for line in lines:
        stripped = line.strip()

        # Detect section headers (Supporting Evidence, Contradictory Evidence, etc.)
        if re.match(
            r"^\*?\*?(supporting|contradictory|related)\s", stripped, re.IGNORECASE
        ):
            in_evidence_section = True
            bullet_count = 0
            result_lines.append(line)
        elif stripped.startswith("-") and in_evidence_section:
            bullet_count += 1
            if bullet_count <= max_papers:
                result_lines.append(line)
        elif stripped and not stripped.startswith("-"):
            # Non-bullet, non-empty line - might be a new section
            in_evidence_section = False
            bullet_count = 0
            result_lines.append(line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def export_gpt_to_csv(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to GPT graph evaluation result file (result.json.zst).",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output CSV file path."),
    ],
    arxiv_cache: Annotated[
        Path,
        typer.Option(
            "--arxiv-cache",
            help="Path to arxiv ID cache file.",
        ),
    ] = ARXIV_CACHE_FILE,
    skip_arxiv: Annotated[
        bool,
        typer.Option(
            "--skip-arxiv",
            help="Skip arxiv lookups (leave PDF links empty).",
        ),
    ] = False,
) -> None:
    """Export GPT graph evaluation results to CSV for annotation pipelines.

    The input file should be the result.json.zst from a GPT graph evaluation run
    (e.g., from `paper gpt eval-graph`). The output CSV contains paper metadata,
    ground truth, predictions, and graph components suitable for human annotation.

    Arxiv IDs are looked up by title and cached to avoid repeated API calls.
    Use --skip-arxiv to skip lookups if PDF links aren't needed.
    """
    setup_logging()

    logger.info(f"Loading GPT graph results from {input_file}")
    gpt_data = load_data(input_file, PromptResult[dict[str, Any]])
    logger.info(f"Loaded {len(gpt_data)} papers")

    # Filter out papers with invalid graphs or rationales
    valid_data = [
        item
        for item in gpt_data
        if _has_valid_graph(item.item["graph"])
        and _has_valid_rationales(item.item["paper"])
    ]
    skipped = len(gpt_data) - len(valid_data)
    if skipped > 0:
        logger.info(f"Skipped {skipped} papers with invalid graphs or rationales")
    gpt_data = valid_data

    # Extract titles and look up arxiv IDs
    titles = [item.item["paper"]["title"] for item in gpt_data]

    if skip_arxiv:
        arxiv_ids: dict[str, str] = {}
        logger.info("Skipping arxiv lookups")
    else:
        arxiv_ids = _lookup_arxiv_ids(titles, arxiv_cache)

    # Build CSV rows
    rows: list[dict[str, str | int]] = []
    for item in gpt_data:
        paper = item.item["paper"]
        graph = item.item["graph"]

        title = paper["title"]
        authors = paper.get("authors", [])
        arxiv_id = arxiv_ids.get(_normalise_title(title), "")
        predicted_rationale = _truncate_evidence_sections(paper["rationale_pred"])

        row = {
            "title": title,
            "authors": _format_authors(authors),
            "pdf_link": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
            "abstract": paper["abstract"],
            "ground_truth_review": paper.get("rationale", ""),
            "ground_truth_rating": paper["rating"],
            "predicted_rationale": predicted_rationale,
            "predicted_rating": paper["y_pred"],
            "claims": _format_entities_as_bullets(graph["entities"], "claim"),
            "methods": _format_entities_as_bullets(graph["entities"], "method"),
            "experiments": _format_entities_as_bullets(graph["entities"], "experiment"),
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "title",
        "authors",
        "pdf_link",
        "abstract",
        "ground_truth_review",
        "ground_truth_rating",
        "predicted_rationale",
        "predicted_rating",
        "claims",
        "methods",
        "experiments",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    papers_with_arxiv = sum(1 for r in rows if r["pdf_link"])
    logger.info(f"Wrote {len(rows)} rows to {output_file}")
    logger.info(f"Papers with arxiv links: {papers_with_arxiv}/{len(rows)}")
