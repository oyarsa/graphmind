"""Merge paper content and review JSON files into a single JSON file.

Only keep those entries where all reviews have a rating.

The original JSON files had some encoding issues, so this sanitises the text to be valid
UTF-8.

The files must be downloaded from Google Drive. See README.md for more information.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _sanitize_value(val: dict[str, Any] | list[Any] | str | Any) -> Any:
    """Sanitise strings to be valid UTF-8, replacing invalid characters with '?'"""
    if isinstance(val, dict):
        return {_sanitize_value(k): _sanitize_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_sanitize_value(v) for v in val]
    elif isinstance(val, str):
        return val.encode("utf-8", errors="replace").decode("utf-8")
    else:
        return val


def _safe_load_json(file_path: Path) -> Any:
    """Load a JSON file, ensuring the text is valid UTF-8."""
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return _sanitize_value(json.load(f))


def merge_content_review(path: Path, output_path: Path) -> None:
    output: list[dict[str, Any]] = []

    for dir in path.iterdir():
        if not dir.is_dir():
            continue

        contents = dir / f"{dir.name}_content"
        reviews = dir / f"{dir.name}_review"

        if not contents.exists() or not reviews.exists():
            continue

        for content_file in contents.glob("*.json"):
            review_file = reviews / content_file.name.replace("_content", "_review")
            if not review_file.exists():
                continue

            content = _safe_load_json(content_file)["metadata"]
            review = _safe_load_json(review_file)["reviews"]

            # We only want entries that have ratings in their reviews and titles
            if all("rating" in r for r in review) and content.get("title"):
                output.append({"paper": content, "review": review, "source": dir.name})

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data",
        type=Path,
        help="Path to directories containing files to merge (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.json",
        type=Path,
        help="Output JSON file (default: %(default)s)",
    )
    args = parser.parse_args()
    merge_content_review(args.path, args.output)


if __name__ == "__main__":
    main()
