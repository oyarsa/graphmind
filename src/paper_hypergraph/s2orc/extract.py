"""Decompress gzipped JSON Lines files, extract data, and save as gzipped JSON.

Extracts abstract, title, venue and paper text. If any of these fields are missing,
the paper is skipped.
"""

import argparse
import contextlib
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm


def extract_annotation(text: str, annotations: dict[str, str], key: str) -> str | None:
    annotation_idxs_str = annotations.get(key)
    if not annotation_idxs_str:
        return None

    try:
        annotation_idxs = json.loads(annotation_idxs_str)
    except json.JSONDecodeError:
        return None

    output: list[str] = []

    for idx in annotation_idxs:
        start = int(idx["start"])
        end = int(idx["end"])
        with contextlib.suppress(IndexError):
            output.append(text[start:end])

    return "\n".join(output)


ANNOTATION_KEYS = ("abstract", "title", "venue")


def process_file(file_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    with gzip.open(file_path, "rt") as f:
        for line in f:
            data = json.loads(line)

            if (
                "content" not in data
                or "text" not in data["content"]
                or "annotations" not in data["content"]
                or not all(
                    data["content"]["annotations"].get(key) for key in ANNOTATION_KEYS
                )
            ):
                continue

            text = data["content"]["text"]
            annotations = data["content"]["annotations"]

            results.append(
                {
                    "abstract": extract_annotation(text, annotations, "abstract"),
                    "title": extract_annotation(text, annotations, "title"),
                    "venue": extract_annotation(text, annotations, "venue"),
                    "text": text,
                }
            )

    return results


def extract_data(files: Iterable[Path]) -> None:
    for file_path in tqdm(tuple(files)):
        try:
            processed = process_file(file_path)
        except Exception as e:
            print(f"ERROR | {file_path} | {e}")
        else:
            output_path = file_path.with_suffix(".json.gz")
            with gzip.open(output_path, "wt") as f:
                json.dump(processed, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="Input gzipped files")
    args = parser.parse_args()
    extract_data(args.files)


if __name__ == "__main__":
    main()
