"""Decompress gzipped JSON Lines files, extract data, and save as gzipped JSON.

Extracts abstract, title, venue and paper text. If any of these fields are missing,
the paper is skipped.
"""

import contextlib
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm


def _extract_annotation(text: str, annotations: dict[str, str], key: str) -> str | None:
    """Extract annotation segment from text using start and end indices."""
    annotation_idxs = annotations.get(key)
    if not annotation_idxs:
        return None

    try:
        annotation_idxs = json.loads(annotation_idxs)
    except json.JSONDecodeError:
        return None

    output: list[str] = []

    for idx in annotation_idxs:
        start = int(idx["start"])
        end = int(idx["end"])
        with contextlib.suppress(IndexError):
            output.append(text[start:end])

    return "\n".join(output)


_ANNOTATION_KEYS = ("abstract", "title", "venue")


def _process_file(file_path: Path) -> list[dict[str, Any]]:
    """Process a single file and return the extracted data.

    Items that are missing the paper content, title or annotations are skipped.
    """
    results: list[dict[str, Any]] = []

    with gzip.open(file_path, "rt") as f:
        for line in f:
            data = json.loads(line)

            if (
                "content" not in data
                or "text" not in data["content"]
                or "annotations" not in data["content"]
                or not all(
                    data["content"]["annotations"].get(key) for key in _ANNOTATION_KEYS
                )
            ):
                continue

            text = data["content"]["text"]
            annotations = data["content"]["annotations"]

            results.append(
                {
                    "abstract": _extract_annotation(text, annotations, "abstract"),
                    "title": _extract_annotation(text, annotations, "title"),
                    "venue": _extract_annotation(text, annotations, "venue"),
                    "text": text,
                }
            )

    return results


def extract_data(input_files: Iterable[Path], output_dir: Path) -> None:
    """Extract the .gz JSON Lines files to JSON.GZ files.

    Only keep those that contain the title and annotations (e.g. abstract, venue, text).

    Args:
        input_files: Input gzipped files to process
        output_dir: Directory where processed files will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(tuple(input_files)):
        try:
            processed = _process_file(file_path)
        except Exception as e:
            print(f"ERROR | {file_path} | {e}")
        else:
            output_path = output_dir / f"{file_path.stem}.json.gz"
            with gzip.open(output_path, "wt") as f:
                json.dump(processed, f)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_files: Annotated[list[Path], typer.Argument(help="Input gzipped files.")],
    output_dir: Annotated[
        Path, typer.Option(help="Directory where processed files will be saved")
    ],
) -> None:
    extract_data(input_files, output_dir)


if __name__ == "__main__":
    app()
