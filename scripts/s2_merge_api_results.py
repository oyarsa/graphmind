"""Add original paper title to Semantic Scholar API results.

The original version of `paper.s2orc.retrieve_papers_semantic_scholar` did not
include the input title query in the output file, which was necessary to evaluate the
match accuracy. This script adds these titles.

Note: the correct version of the API script is already available, so this is a one-off.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    titles_file: Annotated[Path, typer.Argument(help="Path to ASAP Titles JSON file.")],
    api_result_file: Annotated[
        Path, typer.Argument(help="Path to Semantic Scholar API results JSON file.")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output merged JSON file.")
    ],
) -> None:
    titles = json.loads(titles_file.read_text())
    api_result = json.loads(api_result_file.read_text())

    assert len(titles) == len(
        api_result
    ), "Titles and API result must have the same length"

    output = [
        {"title_query": title["title"]} | ({} if result is None else result)
        for title, result in zip(titles, api_result)
    ]

    output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
