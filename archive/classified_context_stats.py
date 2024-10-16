"""Calculate frequency of context polarities and types."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper_hypergraph.gpt.classify_contexts import PaperOutput as Paper

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[Path, typer.Argument(help="Path to input JSON file.")],
) -> None:
    input_data = TypeAdapter(list[Paper]).validate_json(input_file.read_bytes())

    context_polarity: list[str] = []
    context_type: list[str] = []

    for paper in input_data:
        for reference in paper.references:
            for context in reference.contexts:
                context_polarity.append(context.polarity)
                context_type.append(context.type)

    counter_polarity = Counter(context_polarity)
    counter_type = Counter(context_type)

    for member, counter in [("polarity", counter_polarity), ("type", counter_type)]:
        print(">>>", member.upper())

        for key, count in counter.most_common():
            print(f"  {key}: {count} ({count / counter.total():.2%})")

        print()


if __name__ == "__main__":
    app()
