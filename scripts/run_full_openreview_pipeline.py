"""Run the full PETER pipeline from PeerRead preprocessing to graph building.

We reuse existing files if possible. If you want a clean slate, use the `--force` option.
Note that this doesn't re-download the raw PeerRead dataset because it's too large. If
you want to re-download it, manually remove the directory.
"""

import shutil
from pathlib import Path
from typing import Annotated

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
def main(
    input_dir: Annotated[
        Path, typer.Option("--input", "-i", help="Path to the OpenReview dataset.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Directory where the generated file will be saved."
        ),
    ],
    force: Annotated[
        bool, typer.Option(help="Discard existing generated files.")
    ] = False,
    num_papers: Annotated[
        int,
        typer.Option(help="Number of papers from the dataset. Use 0 for all items."),
    ] = 0,
    construct_count: Annotated[
        int,
        typer.Option(
            "--construct",
            help="Number of items for constructed dataset. Use 0 for all items.",
        ),
    ] = 0,
    num_related: Annotated[
        int,
        typer.Option(
            "--related", help="Number of related papers for PETER (each type)."
        ),
    ] = 2,
) -> None:
    """Run the full PETER pipeline from PeerRead preprocessing to graph building."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        shutil.rmtree(output_dir)

    title("Preprocess")
    processed = output_dir / "openreview_merged.json"
    _checkrun(
        processed,
        "openreview",
        "preprocess",
        "--input",
        input_dir,
        "--output",
        processed,
        "--num-papers",
        num_papers,
    )
    assert processed.exists()

    title("Info main")
    info_main_dir = output_dir / "s2_info_main"
    info_main = info_main_dir / "final.json"
    _checkrun(
        info_main,
        "src/paper/semantic_scholar/info.py",
        "main",
        processed,
        info_main_dir,
    )
    assert info_main.exists()

    title("Info references")
    info_ref_dir = output_dir / "s2_info_references"
    info_ref = info_ref_dir / "final.json"
    _checkrun(
        info_ref,
        "src/paper/semantic_scholar/info.py",
        "references",
        processed,
        info_ref_dir,
    )
    assert info_ref.exists()

    title("Recommended")
    recommended_dir = output_dir / "s2_recommended"
    recommended = recommended_dir / "papers_recommended.json"
    _checkrun(
        recommended,
        "src/paper/semantic_scholar/recommended.py",
        info_main,
        recommended_dir,
    )
    assert recommended.exists()

    title("Construct dataset")
    subset_dir = output_dir / "subset"
    peer_with_ref = subset_dir / "peerread_with_s2_references.json"
    peer_related = subset_dir / "peerread_related.json"
    _checkrun(
        peer_with_ref,
        "src/paper/construct_dataset.py",
        "--peerread",
        processed,
        "--references",
        info_ref,
        "--recommended",
        recommended,
        "--output",
        subset_dir,
        "--num-peerread",
        construct_count,
    )
    assert peer_with_ref.exists()

    title("Context")
    context_dir = output_dir / "context"
    context = context_dir / "results.json"
    _checkrun(
        context,
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
    assert context.exists()

    title("PeerRead terms")
    peer_terms_dir = output_dir / "peerread-terms"
    peer_terms = peer_terms_dir / "results_valid.json"
    _checkrun(
        peer_terms,
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
    assert peer_terms.exists()

    title("S2 terms")
    s2_terms_dir = output_dir / "s2-terms"
    s2_terms = s2_terms_dir / "results_valid.json"
    _checkrun(
        s2_terms,
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
    assert s2_terms.exists()

    title("Peter Build")
    peter_graph = output_dir / "peter_graph.json"
    _checkrun(
        peter_graph,
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
    _checkrun(
        peter_peer,
        "peter",
        "peerread",
        "--graph",
        peter_graph,
        "--peerread-ann",
        peer_terms,
        "--num-citations",
        num_related,
        "--num-semantic",
        num_related,
        "--output",
        peter_peer,
    )
    assert peter_peer.exists()

    title("GPT PETER summarisation")
    petersum_dir = output_dir / "peter_summarised"
    petersum = petersum_dir / "result.json"
    _checkrun(
        petersum,
        "gpt",
        "petersum",
        "run",
        "--ann-graph",
        peter_peer,
        "--output",
        petersum_dir,
        "--limit",
        0,
    )
    assert petersum.exists()


def _checkrun(path: Path, *cmd: object) -> None:
    """Run command only if `path` does not already exist.

    Args:
        path: Path to check. The command won't run if it exists. This should be the path
            where a file or directory generated by the command will be.
        *cmd: The command and arguments to run.
    """
    if path.exists():
        print(f"{path} already exists.")
        return

    print(f"{path} does not exist. Running command.")
    run(*cmd)


if __name__ == "__main__":
    app()
