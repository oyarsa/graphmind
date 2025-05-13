from pathlib import Path
import pytest


def assertpath(path: Path) -> None:
    """Assert that `path` exists and print it if it doesn't."""
    if path.exists():
        return

    pytest.fail(f"Path doesn't exist: '{path}'")
