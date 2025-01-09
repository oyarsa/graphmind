"""Run the full PETER pipeline from PeerRead preprocessing to graph building."""

import shutil
from pathlib import Path

import typer

from paper.util.cmd import run, title

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(input_dir: Path, output_dir: Path) -> None:
    """Test the full PETER pipeline from PeerRead preprocessing to graph building."""
    title("Check if PeerRead is available")
    if not input_dir.exists():
        run("src/paper/peerread/download.py", input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir)

    title("Preprocess")
    processed = output_dir / "peerread_merged.json"
    run("preprocess", "peerread", input_dir, processed)
    assert processed.exists()

    title("Info main")
    info_main_dir = output_dir / "s2_info_main"
    run(
        "src/paper/semantic_scholar/info.py",
        "main",
        processed,
        info_main_dir,
    )
    info_main = info_main_dir / "final.json"
    assert info_main.exists()

    title("Info references")
    info_ref_dir = output_dir / "s2_info_references"
    run(
        "src/paper/semantic_scholar/info.py",
        "references",
        processed,
        info_ref_dir,
    )
    info_ref = info_ref_dir / "final.json"
    assert info_ref.exists()

    title("Info areas")
    s2_areas = output_dir / "s2_areas.json"
    run(
        "src/paper/semantic_scholar/areas.py",
        s2_areas,
        "--years",
        "2013-2017",
        "--limit-year",
        "0",
    )
    assert s2_areas.exists()

    title("Recommended")
    recommended_dir = output_dir / "s2_recommended"
    run("src/paper/semantic_scholar/recommended.py", info_main, recommended_dir)
    recommended = recommended_dir / "papers_recommended.json"
    assert recommended.exists()

    title("Construct dataset")
    subset_dir = output_dir / "subset"
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
        "--num-peerrad",
        50,
    )
    peer_with_ref = subset_dir / "peerread_with_s2_references.json"
    peer_related = subset_dir / "peerread_related.json"
    assert peer_with_ref.exists()

    context_dir = output_dir / "context"
    run(
        "gpt",
        "context",
        "run",
        peer_with_ref,
        context_dir,
        "--model",
        "gpt-4o-mini",
        "--limit",
        "0",
    )
    context = context_dir / "result.json"
    assert context.exists()

    peer_terms_dir = output_dir / "peerread-terms"
    run(
        "gpt",
        "terms",
        "run",
        peer_with_ref,
        peer_terms_dir,
        "--paper-type",
        "peerread",
        "--limit",
        "0",
    )
    peer_terms = peer_terms_dir / "results_valid.json"
    assert peer_terms.exists()

    s2_terms_dir = output_dir / "s2-terms"
    run(
        "gpt",
        "terms",
        "run",
        peer_related,
        s2_terms_dir,
        "--paper-type",
        "s2",
        "--limit",
        "0",
    )
    s2_terms = s2_terms_dir / "results_valid.json"
    assert s2_terms.exists()

    title("Peter Build")
    peter_graph = output_dir / "peter_graph.json"
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

    peter_peer = output_dir / "peerread_with_peter.json"
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
    eval_full_dir = output_dir / "eval-full"
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


if __name__ == "__main__":
    app()
