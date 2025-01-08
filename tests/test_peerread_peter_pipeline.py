"""Test the full PETER pipeline from PeerRead preprocessing to graph building."""

from pathlib import Path

import pytest

from .utils import ROOT_DIR, run, run_parallel_commands, title


@pytest.mark.slow
def test_peerread_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from PeerRead preprocessing to graph building."""
    raw_path = ROOT_DIR / "data/PeerRead"

    title("Check if PeerRead is available")
    if not raw_path.exists():
        run("src/paper/peerread/download.py", raw_path)

    title("Preprocess")
    processed_path = tmp_path / "peerread_merged.json"
    run("preprocess", "peerread", raw_path, processed_path, "-n", 100)

    title("Info main")
    run(
        "src/paper/semantic_scholar/info.py",
        "main",
        processed_path,
        tmp_path / "s2_info_main",
        "--limit",
        "1",
    )

    title("Info references")
    run(
        "src/paper/semantic_scholar/info.py",
        "references",
        processed_path,
        tmp_path / "s2_info_references",
        "--limit",
        "1",
    )

    title("Info areas")
    run(
        "src/paper/semantic_scholar/areas.py",
        tmp_path / "s2_areas.json",
        "--years",
        "2020",
        "--limit-year",
        "2",
        "--limit-page",
        "2",
        "--limit-areas",
        "2",
    )

    title("Recommended")
    run(
        "src/paper/semantic_scholar/recommended.py",
        tmp_path / "s2_info_main/valid.json",
        tmp_path / "s2_recommended",
    )

    title("Construct dataset")
    run(
        "src/paper/construct_dataset.py",
        "--peerread",
        processed_path,
        "--references",
        tmp_path / "s2_info_references/final.json",
        "--recommended",
        str(tmp_path / "s2_recommended/papers_recommended.json"),
        "--output",
        str(tmp_path),
    )

    run_parallel_commands(
        [
            (
                "gpt",
                "context",
                "run",
                tmp_path / "peerread_with_s2_references.json",
                tmp_path / "context",
                "--model",
                "gpt-4o-mini",
                "--limit",
                "10",
            ),
            (
                "gpt",
                "terms",
                "run",
                tmp_path / "peerread_with_s2_references.json",
                tmp_path / "peerread-terms",
                "--paper-type",
                "peerread",
                "--limit",
                "10",
            ),
            (
                "gpt",
                "terms",
                "run",
                tmp_path / "peerread_related.json",
                tmp_path / "s2-terms",
                "--paper-type",
                "s2",
                "--limit",
                "10",
            ),
        ]
    )

    title("Peter Build")
    run(
        "peter",
        "build",
        "--ann",
        tmp_path / "s2-terms/results_valid.json",
        "--context",
        tmp_path / "context/result.json",
        "--output",
        tmp_path / "peter_graph.json",
    )

    title("Peter PeerRead")
    run(
        "peter",
        "peerread",
        "--graph",
        tmp_path / "peter_graph.json",
        "--peerread-ann",
        tmp_path / "peerread-terms/results_valid.json",
        "--output",
        tmp_path / "peerread_with_peter.json",
    )

    title("Verify outputs exist")
    assert (tmp_path / "peerread_with_peter.json").exists()
    assert (tmp_path / "peter_graph.json").exists()
    assert (tmp_path / "context/result.json").exists()
    assert (tmp_path / "s2-terms/results_valid.json").exists()
    assert (tmp_path / "peerread-terms/results_valid.json").exists()
