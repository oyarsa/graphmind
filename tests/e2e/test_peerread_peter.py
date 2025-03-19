"""Test the PETER pipeline from PeerRead preprocessing to graph building end-to-end."""

from pathlib import Path

import pytest

from paper.util import git_root
from paper.util.cmd import run, run_parallel_commands, title

ROOT_DIR = git_root()


@pytest.mark.slow
def test_peerread_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from PeerRead preprocessing to graph building."""
    raw_path = ROOT_DIR / "data/PeerRead"

    title("Check if PeerRead is available")
    if not raw_path.exists():
        run("paper", "peerread", "download", raw_path)

    title("Preprocess")
    processed = tmp_path / "peerread_merged.json"
    run("paper", "peerread", "preprocess", raw_path, processed, "-n", 100)
    assert processed.exists()

    title("Info main")
    info_main_dir = tmp_path / "s2_info_main"
    run(
        "paper",
        "s2",
        "info",
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
        "paper",
        "s2",
        "info",
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
        "paper",
        "s2",
        "areas",
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
    run("paper", "s2", "recommended", info_main, recommended_dir)
    recommended = recommended_dir / "papers_recommended.json"
    assert recommended.exists()

    title("Construct dataset")
    subset_dir = tmp_path / "subset"
    run(
        "paper",
        "construct",
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
                "paper",
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
                "paper",
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
                "paper",
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
    context = context_dir / "results.json"
    s2_terms = s2_terms_dir / "results_valid.json"
    peer_terms = peer_terms_dir / "results_valid.json"

    assert context.exists()
    assert s2_terms.exists()
    assert peer_terms.exists()

    title("Peter Build")
    peter_graph = tmp_path / "peter_graph.json"
    run(
        "paper",
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
        "paper",
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

    title("GPT PETER summarisation")
    petersum_dir = tmp_path / "peter_summarised"
    run(
        "paper",
        "gpt",
        "petersum",
        "run",
        "--ann-graph",
        peter_peer,
        "--output",
        petersum_dir,
    )
    petersum = petersum_dir / "result.json"
    assert petersum.exists()

    title("GPT eval sans")
    eval_sans_dir = tmp_path / "eval-sans"
    run(
        "paper",
        "gpt",
        "eval",
        "sans",
        "run",
        "--papers",
        peer_with_ref,
        "--output",
        eval_sans_dir,
        "--demos",
        "eval_4",
    )
    eval_sans = eval_sans_dir / "result.json"
    assert eval_sans.exists()

    title("GPT eval PETER")
    eval_peter_dir = tmp_path / "eval-peter"
    eval_peter = eval_peter_dir / "result.json"
    run(
        "paper",
        "gpt",
        "eval",
        "peter",
        "run",
        "--papers",
        petersum,
        "--output",
        eval_peter_dir,
        "--demos",
        "eval_4",
    )
    assert eval_peter.exists()

    title("Extract ACUs")
    acu_s2_dir = tmp_path / "acu-s2"
    acu_peerread_dir = tmp_path / "acu-peerread"
    run_parallel_commands(
        [
            (
                "paper",
                "gpt",
                "acus",
                "run",
                "--related",
                peer_related,
                "--output",
                acu_s2_dir,
                "--paper-type",
                "s2",
                "--limit",
                "10",
            ),
            (
                "paper",
                "gpt",
                "acus",
                "run",
                "--related",
                peer_with_ref,
                "--output",
                acu_peerread_dir,
                "--paper-type",
                "peerread",
                "--limit",
                "10",
            ),
        ]
    )
    acu_s2 = acu_s2_dir / "results.json"
    acu_peerread = acu_peerread_dir / "results.json"
    assert acu_s2.exists()
    assert acu_peerread.exists()
