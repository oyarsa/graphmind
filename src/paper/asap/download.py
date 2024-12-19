"""Download ASAP dataset from Google Drive."""

# pyright: basic
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import gdown
import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

ASAP_DATSET_FILE_ID = "1nJdljy468roUcKLbVwWUhMs7teirah75"


@app.command(help=__doc__, no_args_is_help=True)
def main(
    output_dir: Annotated[
        Path, typer.Argument(help="Directory to extract the dataset contents")
    ],
) -> None:
    """Download ASAP dataset from Google Drive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        tempfile.TemporaryDirectory() as temp_dir,
        tempfile.NamedTemporaryFile() as temp_zip,
    ):
        url = f"https://drive.google.com/uc?id={ASAP_DATSET_FILE_ID}"
        gdown.download(url, temp_zip.name, quiet=False)

        print("\nExtracting dataset files...")
        with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
            dataset_files = [
                f
                for f in zip_ref.namelist()
                if f.startswith("dataset/") and "__MACOSX" not in f
            ]

            # Extract each file, removing the "dataset/" prefix but keeping subdirs
            for file in dataset_files:
                # Get the relative path without the "dataset/" prefix
                target_path = output_dir / Path(file).relative_to("dataset")
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Only extract if it's a file (not a directory entry)
                if not file.endswith("/"):
                    temp_path = Path(temp_dir)
                    zip_ref.extract(file, temp_path)
                    source = temp_path / file
                    shutil.move(source, target_path)

    print("Download and extraction completed successfully!")


if __name__ == "__main__":
    app()
