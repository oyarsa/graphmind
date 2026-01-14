r"""Filter baseline datasets to match test set papers.

WHY THIS SCRIPT EXISTS
----------------------
The baseline preprocessing pipelines (Novascore ACU extraction, Scimon graph
building) were run on large datasets (~6700 ORC papers, ~170 PeerRead papers)
at different times than the ablation test sets were created. As a result:

- ORC test set (dev_100_balanced): 100 papers from venus4 split
- ORC ACU data: 6737 papers from venus5 pipeline
- ORC Scimon data: 6729 papers from venus5 pipeline

While all 100 test papers ARE present in the full baseline datasets (verified
by title matching), we need to extract exactly those papers to ensure:
1. Fair comparison - same papers evaluated across all methods
2. Comparable variance - same sample size for statistical comparison
3. Reproducibility - documented test set alignment

This script bridges the gap by filtering the large baseline datasets down to
the exact papers in our test sets, using paper titles as the join key.

USAGE
-----
    uv run python scripts/filter_baseline_data.py \\
        --test-set output/venus5/split/dev_100_balanced.json.zst \\
        --input output/venus5/output/acu-query/result.jsonl/result.jsonl \\
        --output output/baselines/orc_acu_100.jsonl \\
        --format acu

TITLE PATHS BY FORMAT
---------------------
- Test set (GraphMind): .item.paper.paper.title
- ACU JSONL (Novascore): .paper.paper.title
- Scimon JSON: .ann.paper.title
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
import zstandard as zstd


class DataFormat(StrEnum):
    """Supported data formats for filtering."""

    ACU = "acu"
    SCIMON = "scimon"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON or compressed JSON file."""
    if path.suffix == ".zst":
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            content = dctx.decompress(f.read())
            return json.loads(content)
    else:
        with open(path) as f:
            return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file (one JSON object per line)."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    """Save JSON or compressed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zst":
        cctx = zstd.ZstdCompressor()
        content = json.dumps(data).encode()
        with open(path, "wb") as f:
            f.write(cctx.compress(content))
    else:
        with open(path, "w") as f:
            json.dump(data, f)


def save_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    """Save JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(item) + "\n" for item in data)


def get_title_from_test_set(item: dict[str, Any]) -> str:
    """Extract title from test set item (GraphMind format)."""
    return item["item"]["paper"]["paper"]["title"]


def get_title_from_acu(item: dict[str, Any]) -> str:
    """Extract title from ACU query result (Novascore format)."""
    return item["paper"]["paper"]["title"]


def get_title_from_scimon(item: dict[str, Any]) -> str:
    """Extract title from Scimon query result."""
    return item["ann"]["paper"]["title"]


app = typer.Typer(help=__doc__)


@app.command()
def main(
    test_set: Annotated[
        Path,
        typer.Option(
            "--test-set",
            help="Path to test set JSON file (GraphMind format)",
        ),
    ],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Path to input baseline data file",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Path to output filtered data file",
        ),
    ],
    data_format: Annotated[
        DataFormat,
        typer.Option(
            "--format",
            help="Format of input data: 'acu' for Novascore JSONL, 'scimon' for Scimon JSON",
        ),
    ],
) -> None:
    """Filter baseline data to match test set papers by title."""
    # Load test set and extract titles
    logger.info(f"Loading test set from {test_set}")
    test_data = load_json(test_set)
    test_titles = {get_title_from_test_set(item) for item in test_data}
    logger.info(f"Found {len(test_titles)} unique titles in test set")

    # Load input data based on format
    logger.info(f"Loading input data from {input_path}")
    if data_format == DataFormat.ACU:
        input_data = load_jsonl(input_path)
        get_title = get_title_from_acu
    else:
        input_data = load_json(input_path)
        get_title = get_title_from_scimon

    logger.info(f"Loaded {len(input_data)} items from input")

    # Filter by title match
    filtered = [item for item in input_data if get_title(item) in test_titles]
    logger.info(f"Filtered to {len(filtered)} items matching test set")

    # Report any missing papers
    filtered_titles = {get_title(item) for item in filtered}
    missing = test_titles - filtered_titles
    if missing:
        logger.warning(f"Missing {len(missing)} papers from input data:")
        for title in sorted(missing)[:5]:
            logger.warning(f"  - {title[:60]}...")
        if len(missing) > 5:
            logger.warning(f"  ... and {len(missing) - 5} more")

    # Save filtered data
    logger.info(f"Saving filtered data to {output_path}")
    if data_format == DataFormat.ACU:
        save_jsonl(filtered, output_path)
    else:
        save_json(filtered, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    app()
