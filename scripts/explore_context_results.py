"""TODO: write docstring."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper_hypergraph.gpt.classify_contexts import PaperOutput, show_classified_stats
from paper_hypergraph.gpt.run_gpt import PromptResult

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to the output file from context classification."),
    ],
) -> None:
    input_data = TypeAdapter(list[PromptResult[PaperOutput]]).validate_json(
        input_file.read_bytes()
    )

    papers = [result.item for result in input_data]
    contexts = [
        context
        for paper in papers
        for reference in paper.references
        for context in reference.contexts
    ]

    print(f"Papers  : {len(papers)}")
    print(f"Contexts: {len(contexts)}")

    info, metrics = show_classified_stats(papers)
    print(info)
    if metrics:
        print(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
