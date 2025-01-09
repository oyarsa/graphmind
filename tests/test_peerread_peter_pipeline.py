"""Test the full PETER pipeline from PeerRead preprocessing to graph building."""

from pathlib import Path

import pytest

from paper.util.cmd import run, run_parallel_commands, title

ROOT_DIR = Path(__file__).parent.parent


@pytest.mark.slow
def test_peerread_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from PeerRead preprocessing to graph building."""
    raw_path = ROOT_DIR / "data/PeerRead"

    title("Check if PeerRead is available")
    if not raw_path.exists():
        run("src/paper/peerread/download.py", raw_path)

    title("Preprocess")
    processed = tmp_path / "peerread_merged.json"
    run("preprocess", "peerread", raw_path, processed, "-n", 100)
    assert processed.exists()

    title("Info main")
    info_main_dir = tmp_path / "s2_info_main"
    run(
        "src/paper/semantic_scholar/info.py",
        "main",
        processed,
        info_main_dir,
        "--limit",
        "1",
    )
    info_main = info_main_dir / "final.json"
    assert info_main.exists()

    title("Info references")
    info_ref_dir = tmp_path / "s2_info_references"
    run(
        "src/paper/semantic_scholar/info.py",
        "references",
        processed,
        info_ref_dir,
        "--limit",
        "1",
    )
    info_ref = info_ref_dir / "final.json"
    assert info_ref.exists()

    title("Info areas")
    s2_areas = tmp_path / "s2_areas.json"
    run(
        "src/paper/semantic_scholar/areas.py",
        s2_areas,
        "--years",
        "2020",
        "--limit-year",
        "2",
        "--limit-page",
        "2",
        "--limit-areas",
        "2",
    )
    assert s2_areas.exists()

    title("Recommended")
    recommended_dir = tmp_path / "s2_recommended"
    run("src/paper/semantic_scholar/recommended.py", info_main, recommended_dir)
    recommended = recommended_dir / "papers_recommended.json"
    assert recommended.exists()

    title("Construct dataset")
    subset_dir = tmp_path / "subset"
    run(
        "src/paper/construct_dataset.py",
        "--peerread",
        processed,
        "--references",
        info_ref,
        "--recommended",
        recommended,
        "--output",
        subset_dir,
    )
    peer_with_ref = subset_dir / "peerread_with_s2_references.json"
    peer_related = subset_dir / "peerread_related.json"
    assert peer_with_ref.exists()

    context_dir = tmp_path / "context"
    peer_terms_dir = tmp_path / "peerread-terms"
    s2_terms_dir = tmp_path / "s2-terms"
    run_parallel_commands(
        [
            (
                "gpt",
                "context",
                "run",
                peer_with_ref,
                context_dir,
                "--model",
                "gpt-4o-mini",
                "--limit",
                "10",
            ),
            (
                "gpt",
                "terms",
                "run",
                peer_with_ref,
                peer_terms_dir,
                "--paper-type",
                "peerread",
                "--limit",
                "10",
            ),
            (
                "gpt",
                "terms",
                "run",
                peer_related,
                s2_terms_dir,
                "--paper-type",
                "s2",
                "--limit",
                "10",
            ),
        ]
    )
    context = context_dir / "result.json"
    s2_terms = s2_terms_dir / "results_valid.json"
    peer_terms = peer_terms_dir / "results_valid.json"

    assert context.exists()
    assert s2_terms.exists()
    assert peer_terms.exists()

    title("Peter Build")
    peter_graph = tmp_path / "peter_graph.json"
    run(
        "peter",
        "build",
        "--ann",
        s2_terms,
        "--context",
        context,
        "--output",
        peter_graph,
    )
    assert peter_graph.exists()

    peter_peer = tmp_path / "peerread_with_peter.json"
    title("Peter PeerRead")
    run(
        "peter",
        "peerread",
        "--graph",
        peter_graph,
        "--peerread-ann",
        peer_terms,
        "--output",
        peter_peer,
    )
    assert peter_peer.exists()

    title("GPT eval full")
    eval_full_dir = tmp_path / "eval-full"
    run(
        "gpt",
        "eval",
        "full",
        "run",
        "--peerread",
        processed,
        "--output",
        eval_full_dir,
        "--demos",
        "eval_demonstrations_4",
    )
    eval_full = eval_full_dir / "result.json"
    assert eval_full.exists()
