"""Test the full PETER pipeline from preprocessing to graph building."""

from pathlib import Path

import pytest

from .utils import run, run_parallel_commands


@pytest.mark.slow
def test_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from preprocessing to graph building."""

    # Download ASAP
    run("src/paper/asap/download.py", str(tmp_path / "asap-dataset"))

    # Preprocess
    run("preprocess", "asap", "data/asap", str(tmp_path), "100")

    # Info main
    run(
        "src/paper/semantic_scholar/info.py",
        "main",
        tmp_path / "asap_filtered.json",
        tmp_path / "s2_info_main",
        "--limit",
        "1",
    )

    # Info references
    run(
        "src/paper/semantic_scholar/info.py",
        "references",
        tmp_path / "asap_filtered.json",
        tmp_path / "s2_info_references",
        "--limit",
        "1",
    )

    # Info areas
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

    # Recommended
    run(
        "src/paper/semantic_scholar/recommended.py",
        tmp_path / "s2_info_main/valid.json",
        tmp_path / "s2_recommended",
    )

    # Construct dataset
    run(
        "src/paper/construct_dataset.py",
        "--asap",
        tmp_path / "asap_filtered.json",
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
                tmp_path / "asap_with_s2_references.json",
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
                tmp_path / "asap_with_s2_references.json",
                tmp_path / "asap-terms",
                "--paper-type",
                "asap",
                "--limit",
                "10",
            ),
            (
                "gpt",
                "terms",
                "run",
                tmp_path / "asap_related.json",
                tmp_path / "s2-terms",
                "--paper-type",
                "s2",
                "--limit",
                "10",
            ),
        ]
    )

    # Peter Build
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

    # Peter ASAP
    run(
        "peter",
        "asap",
        "--graph",
        tmp_path / "peter_graph.json",
        "--asap-ann",
        tmp_path / "asap-terms/results_valid.json",
        "--output",
        tmp_path / "asap_with_peter.json",
    )

    # Verify outputs exist
    assert (tmp_path / "asap_with_peter.json").exists()
    assert (tmp_path / "peter_graph.json").exists()
    assert (tmp_path / "context/result.json").exists()
    assert (tmp_path / "s2-terms/results_valid.json").exists()
    assert (tmp_path / "asap-terms/results_valid.json").exists()
