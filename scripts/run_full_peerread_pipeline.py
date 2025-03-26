"""Run the full PeerRead pipeline preprocessing to graph building.

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
        Path, typer.Option("--input", "-i", help="Path to the PeerRead dataset.")
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
    ] = 50,
    num_related: Annotated[
        int,
        typer.Option(
            "--related", help="Number of related papers for PETER (each type)."
        ),
    ] = 2,
    references_top_k: Annotated[
        int,
        typer.Option(
            "--references-k",
            help="How many references to query per paper, sorted by semantic similarity.",
        ),
    ] = 20,
    num_recommendations: Annotated[
        int,
        typer.Option("--recommended", help="Number of recommendations per paper."),
    ] = 30,
) -> None:
    """Run the full PETER pipeline from PeerRead preprocessing to graph building."""
    title("Check if PeerRead is available")
    if not input_dir.exists():
        run("paper", "peerread", "download", input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        shutil.rmtree(output_dir)

    title("Preprocess")
    processed = output_dir / "peerread_merged.json"
    _checkrun(
        processed,
        "paper",
        "peerread",
        "preprocess",
        input_dir,
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
        "paper",
        "s2",
        "info",
        "main",
        processed,
        info_main_dir,
        "--limit",
        0,
    )
    assert info_main.exists()

    title("Info references")
    info_ref_dir = output_dir / "s2_info_references"
    info_ref = info_ref_dir / "final.json"
    _checkrun(
        info_ref,
        "paper",
        "s2",
        "info",
        "references",
        processed,
        info_ref_dir,
        "--limit",
        0,
        "--top-k",
        references_top_k,
    )
    assert info_ref.exists()

    title("Info areas")
    s2_areas = output_dir / "s2_areas.json"
    _checkrun(
        s2_areas,
        "paper",
        "s2",
        "areas",
        s2_areas,
        "--years",
        "2013-2017",
        "--limit-year",
        "0",
    )
    assert s2_areas.exists()

    title("Recommended")
    recommended_dir = output_dir / "s2_recommended"
    recommended = recommended_dir / "papers_recommended.json"
    _checkrun(
        recommended,
        "paper",
        "s2",
        "recommended",
        info_main,
        recommended_dir,
        "--limit-papers",
        0,
        "--limit-recommendations",
        num_recommendations,
    )
    assert recommended.exists()

    title("Construct dataset")
    subset_dir = output_dir / "subset"
    peer_with_ref = subset_dir / "peerread_with_s2_references.json"
    peer_related = subset_dir / "peerread_related.json"
    _checkrun(
        peer_with_ref,
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
        "--num-peerread",
        construct_count,
    )
    assert peer_with_ref.exists()

    title("Context")
    context_dir = output_dir / "context"
    context = context_dir / "results.json"
    _checkrun(
        context,
        "paper",
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
        "paper",
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
        "paper",
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
    peter_graph_dir = output_dir / "peter_graph"
    peter_graph_file = (
        peter_graph_dir / "citation_graph.json"
    )  # We check for this file to exist
    _checkrun(
        peter_graph_file,
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
    assert peter_graph_file.exists()

    peter_peer = output_dir / "peerread_with_peter.json"
    title("Peter PeerRead")
    _checkrun(
        peter_peer,
        "paper",
        "peter",
        "peerread",
        "--graph-dir",
        peter_graph_dir,
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
        "paper",
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

    title("SciMON build")
    scimon_graph_dir = output_dir / "scimon_graph"
    scimon_kg_file = scimon_graph_dir / "kg_graph.json"
    _checkrun(
        scimon_kg_file,
        "paper",
        "baselines",
        "scimon",
        "build",
        "--ann",
        s2_terms,
        "--peerread",
        peer_with_ref,
        "--output-dir",
        scimon_graph_dir,
        "--num-annotated",
        5000,
    )
    assert scimon_kg_file.exists()

    title("SciMON query")
    scimon_peer = output_dir / "peerread_with_scimon.json"
    _checkrun(
        scimon_peer,
        "paper",
        "baselines",
        "scimon",
        "query",
        "--ann-peer",
        peer_terms,
        "--graph-dir",
        scimon_graph_dir,
        "--output",
        scimon_peer,
    )

    title("Extract ACUs S2")
    acu_s2_dir = output_dir / "acu-s2"
    acu_s2 = acu_s2_dir / "result.json"
    _checkrun(
        acu_s2,
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
        "0",
    )
    assert acu_s2.exists()

    title("Extract ACUs PeerRead")
    acu_peerread_dir = output_dir / "acu-peerread"
    acu_peerread = acu_peerread_dir / "result.json"
    _checkrun(
        acu_peerread,
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
        "0",
    )
    assert acu_peerread.exists()

    title("Build Nova database")
    acu_db = output_dir / "acu-db"
    _checkrun(
        acu_db,
        "paper",
        "baselines",
        "nova",
        "build",
        "--input",
        acu_s2,
        "--output",
        acu_db,
        "--limit",
        0,
        "--sentences",
        500_000,
    )
    assert acu_db.exists()

    title("Query Nova database")
    acu_query_dir = output_dir / "acu-query"
    acu_query = acu_query_dir / "result.jsonl"
    _checkrun(
        acu_query,
        "paper",
        "baselines",
        "nova",
        "query",
        "--db",
        acu_db,
        "--input",
        acu_peerread,
        "--output",
        acu_query,
        "--limit",
        0,
    )
    assert acu_query.exists()


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
