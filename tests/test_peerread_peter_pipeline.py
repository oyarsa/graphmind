"""Test the full PETER pipeline from PeerRead preprocessing to graph building."""

from pathlib import Path

import pytest

from .utils import run


@pytest.mark.slow
def test_peer_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from PeerRead preprocessing to graph building."""

    # Download PeerRead
    raw_path = tmp_path / "peerread-dataset"
    run("src/paper/peerread/download.py", raw_path)

    # Preprocess
    processed_path = tmp_path / "peerread_merged.json"
    run("src/paper/peerread/process.py", raw_path, processed_path, "-n", 100)
