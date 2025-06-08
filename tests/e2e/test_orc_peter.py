"""Test the PETER pipeline from ORC preprocessing to graph building end-to-end."""

from pathlib import Path

import pytest

from paper.util import git_root
from paper.util.cmd import run, run_parallel_commands, title
from tests.helpers import assertpath  # type: ignore[reportMissingImports]

ROOT_DIR = git_root()


@pytest.mark.slow
def test_orc_peter_pipeline(tmp_path: Path) -> None:
    """Test the full PETER pipeline from ORC preprocessing to graph building."""
    raw_path = ROOT_DIR / "data/openreview"

    assertpath(raw_path)

    title("Preprocess")
    processed = tmp_path / "orc_merged.json.zst"
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
    assertpath(processed)

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
    info_main = info_main_dir / "final.json.zst"
    assertpath(info_main)

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
    info_ref = info_ref_dir / "final.json.zst"
    assertpath(info_ref)

    title("Info areas")
    s2_areas = tmp_path / "s2_areas.json.zst"
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
    assertpath(s2_areas)

    title("Recommended")
    recommended_dir = tmp_path / "s2_recommended"
    run("paper", "s2", "recommended", info_main, recommended_dir)
    recommended = recommended_dir / "papers_recommended.json.zst"
    assertpath(recommended)

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
    orc_with_ref = subset_dir / "peerread_with_s2_references.json.zst"
    orc_related = subset_dir / "peerread_related.json.zst"
    assertpath(orc_with_ref)

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
    context = context_dir / "results.json.zst"
    s2_terms = s2_terms_dir / "results_valid.json.zst"
    orc_terms = orc_terms_dir / "results_valid.json.zst"

    assertpath(context)
    assertpath(s2_terms)
    assertpath(orc_terms)

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
    assertpath(peter_graph_dir)

    peter_orc = tmp_path / "orc_with_peter.json.zst"
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
    assertpath(peter_orc)

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
    petersum = petersum_dir / "result.json.zst"
    assertpath(petersum)

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
    eval_sans = eval_sans_dir / "result.json.zst"
    assertpath(eval_sans)

    title("GPT eval graph")
    eval_graph_dir = tmp_path / "eval-graph"
    eval_graph = eval_graph_dir / "result.json.zst"
    run(
        "paper",
        "gpt",
        "eval",
        "graph",
        "run",
        "--papers",
        petersum,
        "--output",
        eval_graph_dir,
        "--eval-prompt",
        "related",
        "--demos",
        "eval_4",
    )
    assertpath(eval_graph)

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
                "--input",
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
                "--input",
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
    acu_s2 = acu_s2_dir / "result.json.zst"
    acu_orc = acu_orc_dir / "result.json.zst"
    assertpath(acu_s2)
    assertpath(acu_orc)
