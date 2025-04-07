from pathlib import Path
from paper.util import git_root
import pytest


def assertpath(path: Path) -> None:
    """Assert that `path` exists and print it if it doesn't."""
    if path.exists():
        return

    relative = path.relative_to(git_root())
    pytest.fail(f"Path doesn't exist: '{relative}'")
