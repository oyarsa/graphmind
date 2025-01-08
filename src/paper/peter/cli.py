"""Create PETER graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from paper import embedding as emb
from paper import gpt
from paper import semantic_scholar as s2
from paper.peter import citations, graph, semantic
from paper.util import Timer, display_params, setup_logging
from paper.util.serde import Record, load_data, save_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="peter",
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.callback(
    help=__doc__,
)
def main() -> None:
    """Empty callback for documentation."""
    setup_logging()


@app.command(name="citations", help="Create citations graph.", no_args_is_help=True)
def citations_(
    input_file: Annotated[
        Path,
        typer.Option(
            "--peerread",
            help="File with PeerRead papers with references with full S2 data and"
            " classified contexts.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", help="Citation graph as a JSON file."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    logger.debug("Loading classified papers.")
    peerread_papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperWithContextClassfied])
    )

    logger.debug("Loading encoder.")
    encoder = emb.Encoder(model_name)
    logger.debug("Creating graph.")
    graph = citations.Graph.from_papers(encoder, peerread_papers, progress=True)

    logger.debug("Saving graph.")
    save_data(output_file, graph)


@app.command(name="semantic", help="Create semantic graph.", no_args_is_help=True)
def semantic_(
    input_file: Annotated[
        Path,
        typer.Option(
            "--peerread",
            help="File with PeerRead papers with extracted backgrounds and targets.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", help="Semantic graph as a JSON file."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    logger.debug("Loading annotated papers.")
    peerread_papers = gpt.PromptResult.unwrap(
        load_data(input_file, gpt.PromptResult[gpt.PaperAnnotated])
    )

    logger.debug("Loading encoder.")
    encoder = emb.Encoder(model_name)
    logger.debug("Creating graph.")
    graph = semantic.Graph.from_papers(encoder, peerread_papers, progress=True)

    logger.debug("Saving graph.")
    save_data(output_file, graph.to_data())


@app.command(help="Create full graph.", no_args_is_help=True)
def build(
    ann_file: Annotated[
        Path,
        typer.Option(
            "--ann",
            help="File with PeerRead papers with extracted backgrounds and targets.",
        ),
    ],
    context_file: Annotated[
        Path,
        typer.Option(
            "--context",
            help="File with PeerRead papers with classified contexts.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output", help="Full graph as a JSON file."),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    logger.debug("Loading annotated papers.")
    papers_ann = gpt.PromptResult.unwrap(
        load_data(ann_file, gpt.PromptResult[gpt.PaperAnnotated])
    )
    logger.debug("Loading context papers.")
    papers_context = gpt.PromptResult.unwrap(
        load_data(context_file, gpt.PromptResult[gpt.PaperWithContextClassfied])
    )

    logger.debug("Loading encoder.")
    encoder = emb.Encoder(model_name)

    logger.debug("Building graph.")
    main_graph = graph.Graph.from_papers(encoder, papers_ann, papers_context)

    logger.info("Saving graph.")
    save_data(output_file, main_graph.to_data())


@app.command(no_args_is_help=True)
def query(
    graph_file: Annotated[
        Path, typer.Option("--graph", help="Path to full graph file.")
    ],
    ann_file: Annotated[
        Path,
        typer.Option(
            "--peerread-ann",
            help="File with PeerRead papers with extracted backgrounds and targets.",
        ),
    ],
    titles: Annotated[
        list[str] | None,
        typer.Option(
            help="Title of the paper to test query. If absent, use an arbitrary paper."
        ),
    ] = None,
    num_papers: Annotated[
        int,
        typer.Option(
            "--num-papers",
            "-n",
            help="Number of papers to query if --title isn't given",
        ),
    ] = 1,
) -> None:
    """Demonstrate the graph by querying it to get polarised related papers."""
    logger.info(display_params())

    logger.debug("Loading papers.")
    ann = gpt.PromptResult.unwrap(
        load_data(ann_file, gpt.PromptResult[gpt.PeerReadAnnotated])
    )

    if not titles:
        papers = ann[:num_papers]
    else:
        papers = [
            next(p for p in ann if s2.clean_title(p.title) == s2.clean_title(title))
            for title in titles
        ]

    logger.debug("Loading graph.")
    main_graph = graph.graph_from_json(graph_file)

    for paper in papers:
        logger.debug("Querying graph.")

        with Timer("Graph query") as timer:
            result = main_graph.query_all(paper.id, paper.background, paper.target)
        logger.debug(timer)

        print(paper.title)
        print()
        for label, papers in [
            (">> semantic_positive", result.semantic_positive),
            (">> semantic_negative", result.semantic_negative),
            (">> citations_positive", result.citations_positive),
            (">> citations_negative", result.citations_negative),
        ]:
            print(f"{label} ({len(papers)})")
            for p in papers:
                print(f"- {p.title}")
            print()


@app.command(no_args_is_help=True)
def peerread(
    graph_file: Annotated[
        Path, typer.Option("--graph", help="Path to full graph file.")
    ],
    ann_file: Annotated[
        Path,
        typer.Option(
            "--peerread-ann",
            help="File with PeerRead papers with extracted backgrounds and targets.",
        ),
    ],
    output_file: Annotated[
        Path, typer.Option("--output", help="Output file to save the result data.")
    ],
    num_papers: Annotated[
        int | None,
        typer.Option(
            "--num-papers",
            "-n",
            help="Number of papers to query. Defaults to all papers.",
        ),
    ] = None,
) -> None:
    """Query the graph with PeerRead papers and save the results in a file.

    The output file contains both the original PeerRead paper and the graph query results.
    """
    logger.info(display_params())

    logger.debug("Loading papers.")
    papers = gpt.PromptResult.unwrap(
        load_data(ann_file, gpt.PromptResult[gpt.PeerReadAnnotated])
    )[:num_papers]

    logger.debug("Loading graph.")
    main_graph = graph.graph_from_json(graph_file)

    results = [
        PaperResult(
            paper=paper,
            results=main_graph.query_all(paper.id, paper.background, paper.target),
        )
        for paper in tqdm(papers, desc="Querying PeerRead papers.")
    ]
    save_data(output_file, results)


class PaperResult(Record):
    """PeerRead paper with its related papers queried from the PETER graph."""

    paper: gpt.PeerReadAnnotated
    results: graph.QueryResult

    @property
    def id(self) -> str:
        """Identify graph result as the underlying paper's ID."""
        return self.paper.id
