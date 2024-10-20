"""Compare the number of references before and after reference-abstract matching.

Input files are the outputs of asap/filter and asap/add_reference_abstracts.
"""

import json
from pathlib import Path
from statistics import median, mode
from typing import Annotated, Any

import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command(help=__doc__)
def main(
    filtered_file: Annotated[Path, typer.Argument(help="Path to asap_filtered.json.")],
    abstract_file: Annotated[
        Path, typer.Argument(help="Path to asap_ref_abstracts file.")
    ],
) -> None:
    filtered: list[dict[str, Any]] = json.loads(filtered_file.read_text())
    abstract: list[dict[str, Any]] = json.loads(abstract_file.read_text())
    assert len(filtered) == len(abstract), "Lengths should be the same"

    filtered_all_references = 0
    abstract_all_references = 0
    all_lost_references: list[int] = []
    for f, a in zip(filtered, abstract):
        assert f["title"] == a["title"], "Titles should be the same"
        filtered_all_references += len(f["references"])
        abstract_all_references += len(a["references"])
        all_lost_references.append(len(f["references"]) - len(a["references"]))

    print(f"Avg  references lost: {sum(all_lost_references) / len(filtered):.2f}")
    print(f"Min  references lost: {min(all_lost_references)}")
    print(f"Max  references lost: {max(all_lost_references)}")
    print(f"Med  references lost: {median(all_lost_references)}")
    print(f"Mode references lost: {mode(all_lost_references)}")

    print()
    print(f"Num papers: {len(filtered)}")
    print(f"Num references before: {filtered_all_references}")
    print(f"Numfreferences after : {abstract_all_references}")
    print(f"Avg references before: {filtered_all_references/len(filtered):.2f}")
    print(f"Avg references after : {abstract_all_references/len(abstract):.2f}")


if __name__ == "__main__":
    app()
