"""Try all possible data types and see what works."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from beartype.door import is_bearable
from pydantic import ValidationError

import paper.gpt.annotate_paper
import paper.gpt.classify_contexts
import paper.gpt.evaluate_paper
import paper.gpt.evaluate_tournament.comparisons
import paper.gpt.evaluate_tournament.tournament
import paper.gpt.extract_graph
import paper.gpt.model
import paper.peerread.model
import paper.semantic_scholar.model
from paper.util.serde import get_full_type_name

TYPES = [
    paper.gpt.annotate_paper.AbstractDemonstration,
    paper.gpt.classify_contexts.PaperWithContextClassfied,
    paper.gpt.evaluate_paper.Demonstration,
    paper.gpt.evaluate_paper.PaperResult,
    paper.gpt.extract_graph.ExtractedGraph,
    paper.gpt.extract_graph.GraphResult,
    paper.gpt.model.PaperAnnotated,
    paper.gpt.model.PaperWithACUs,
    paper.gpt.model.PaperWithRelatedSummary,
    paper.gpt.model.PeerReadAnnotated,
    paper.peerread.model.Paper,
    paper.semantic_scholar.model.Paper,
    paper.semantic_scholar.model.PaperArea,
    paper.semantic_scholar.model.PaperFromPeerRead,
    paper.semantic_scholar.model.PaperRecommended,
    paper.semantic_scholar.model.PaperWithS2Refs,
    paper.semantic_scholar.model.PeerReadPaperWithS2,
    paper.gpt.evaluate_tournament.comparisons.RawComparisonOutput,
    paper.gpt.evaluate_tournament.tournament.TournamentSummary,
]

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[Path, typer.Argument(help="File to try to read.")],
) -> None:
    """Try to read `input_file` with all possible types and print what works."""
    data = json.loads(input_file.read_bytes())
    if is_bearable(data, list[Any]):
        item = data[0]
    else:
        item = data

    valid_types: list[str] = []

    for type_ in TYPES:
        for version in [type_, paper.gpt.model.PromptResult[type_]]:
            try:
                version.model_validate(item)
            except ValidationError:
                pass
            else:
                valid_types.append(get_full_type_name(version))

    if not valid_types:
        typer.secho("Could not find any valid types.", err=True, fg=typer.colors.RED)
    else:
        n = len(valid_types)
        suffix = "s" if n != 1 else ""
        typer.secho(f"Found {n} type{suffix}:", err=True, fg=typer.colors.GREEN)

    for type_ in valid_types:
        typer.echo(f"- {type_}")


if __name__ == "__main__":
    app()
