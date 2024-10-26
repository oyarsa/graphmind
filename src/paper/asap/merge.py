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


def _safe_load_json(file_path: Path) -> Any:
    """Load a JSON file, removing invalid UTF-8 characters."""
    return json.loads(file_path.read_text(encoding="utf-8", errors="replace"))


def merge_content_review(path: Path, output_path: Path, max_papers: int | None) -> None:
    """Read papers in `path`, extract content and reviews, and writes to `output_path`.

    Only papers with titles and reviews with ratings are kept.

    This also does some unicode sanitisation, as the original JSON files had some
    encoding issues. This might lead to fallback characters in the output.
    """
    output: list[dict[str, Any]] = []
    count = 0

    for dir in path.iterdir():
        if not dir.is_dir():
            continue

        contents = dir / f"{dir.name}_content"
        reviews = dir / f"{dir.name}_review"
        papers = dir / f"{dir.name}_paper"

        if not contents.exists() or not reviews.exists() or not papers.exists():
            continue

        for content_file in contents.glob("*.json"):
            if max_papers is not None and count >= max_papers:
                break

            review_file = reviews / content_file.name.replace("_content", "_review")
            paper_file = papers / content_file.name.replace("_content", "_paper")

            if not review_file.exists() or not paper_file.exists():
                continue

            content = _safe_load_json(content_file)["metadata"]
            review = _safe_load_json(review_file)["reviews"]
            approval = _safe_load_json(paper_file)["decision"]

            # We only want entries that have ratings in their reviews and titles
            if all("rating" in r for r in review) and content.get("title"):
                output.append(
                    {
                        "paper": content,
                        "review": review,
                        "source": dir.name,
                        "approval": approval,
                    }
                )
                count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to directories containing files to merge",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output merged JSON file",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process",
    )
    args = parser.parse_args()
    merge_content_review(args.path, args.output, args.max_papers)


if __name__ == "__main__":
    main()
