"""Test the PETER pipeline from ORC preprocessing to graph building end-to-end."""

from pathlib import Path

import pytest

from paper.util import git_root
from paper.util.cmd import run, run_parallel_commands, title

ROOT_DIR = git_root()


@pytest.mark.slow
def test_orc_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from ORC preprocessing to graph building."""
    raw_path = ROOT_DIR / "data/openreview"

    assert raw_path.exists()

    title("Preprocess")
    processed = tmp_path / "orc_merged.json"
    run(
        "paper",
        "orc",
        "preprocess",
        "--input",
        raw_path,
        "--output",
        processed,
        "--num-papers",
        100,
    )
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
    orc_with_ref = subset_dir / "peerread_with_s2_references.json"
    orc_related = subset_dir / "peerread_related.json"
    assert orc_with_ref.exists()

    context_dir = tmp_path / "context"
    orc_terms_dir = tmp_path / "orc-terms"
    s2_terms_dir = tmp_path / "s2-terms"
    run_parallel_commands(
        [
            (
                "paper",
                "gpt",
                "context",
                "run",
                orc_with_ref,
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
                orc_with_ref,
                orc_terms_dir,
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
                orc_related,
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
    orc_terms = orc_terms_dir / "results_valid.json"

    assert context.exists()
    assert s2_terms.exists()
    assert orc_terms.exists()

    title("Peter Build")
    peter_graph_dir = tmp_path / "peter_graph"
    run(
        "paper",
        "peter",
        "build",
        "--ann",
        s2_terms,
        "--context",
        context,
        "--output-dir",
        peter_graph_dir,
    )
    assert peter_graph_dir.exists()

    peter_orc = tmp_path / "orc_with_peter.json"
    title("Peter ORC")
    run(
        "paper",
        "peter",
        "peerread",
        "--graph-dir",
        peter_graph_dir,
        "--peerread-ann",
        orc_terms,
        "--output",
        peter_orc,
    )
    assert peter_orc.exists()

    title("GPT PETER summarisation")
    petersum_dir = tmp_path / "peter_summarised"
    run(
        "paper",
        "gpt",
        "petersum",
        "run",
        "--ann-graph",
        peter_orc,
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
        orc_with_ref,
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
    acu_orc_dir = tmp_path / "acu-orc"
    run_parallel_commands(
        [
            (
                "paper",
                "gpt",
                "acus",
                "run",
                "--related",
                orc_related,
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
                orc_with_ref,
                "--output",
                acu_orc_dir,
                "--paper-type",
                "peerread",
                "--limit",
                "10",
            ),
        ]
    )
    acu_s2 = acu_s2_dir / "results.json"
    acu_orc = acu_orc_dir / "results.json"
    assert acu_s2.exists()
    assert acu_orc.exists()
