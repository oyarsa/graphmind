"""Build an index from multiple gz-compressed JSON files containing paper data.

The index is a JSON file containing an object with fields `title` -> `file name`.
The file names are only the basename for the files, assumed to be under
`input_directory`.
"""

import gzip
import sys
from pathlib import Path
from typing import Annotated, Any

import orjson
import typer
from beartype.door import is_bearable
from tqdm import tqdm

from paper.util.serde import write_file_bytes

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_dir: Annotated[
        Path, typer.Argument(help="Path to the directory containing data files.")
    ],
    output_file: Annotated[Path, typer.Argument(help="Path to the output index file.")],
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit on the number of files to process."),
    ] = None,
) -> None:
    """Process S2ORC papers from multiple compressed JSON files and build an index.

    The input data is the output of the paper.s2orc.extract module.
    The index is title -> file. The title is raw from the 'title' field in the output.
    """

    input_files = list(input_dir.glob("*.json.gz"))
    if not input_files:
        raise ValueError(f"No .json.gz files found in {input_dir}")

    title_index: dict[str, str] = {}

    for file_path in tqdm(input_files[:limit]):
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as infile:
                data = orjson.loads(infile.read())
                if not is_bearable(data, list[dict[str, Any]]):
                    print(f"Data file has invalid format: {file_path}")
                    continue

                for paper in data:
                    if title := paper.get("title"):
                        title_index[title] = file_path.name

        except (orjson.JSONDecodeError, OSError) as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_file_bytes(output_file, orjson.dumps(title_index))


if __name__ == "__main__":
    app()
