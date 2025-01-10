"""Download the PeerRead dataset from the repository under a fixed commit.

WARNING: downloads about 2 GB of data.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

REPOSITORY = "https://github.com/allenai/PeerRead"
COMMIT = "9bb37751781a900cee9e74ec3105997732c8e8e5"


def download(
    output_dir: Annotated[Path, typer.Argument(help="Path to save the PeerRead data.")],
) -> None:
    """Fetch the specified commit and copy its data directory to the output directory."""
    print(f"Repository: {REPOSITORY}")
    print(f"Commit: {COMMIT}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", REPOSITORY],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", COMMIT],
            cwd=repo_path,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", COMMIT],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        data_dir = repo_path / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Data directory not found in repository: {data_dir}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(data_dir, output_dir, dirs_exist_ok=True)
