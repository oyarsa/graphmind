"""Diff between two prompt texts (`gpt.prompts.PromptTemplate`)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
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


@app.command(help=__doc__)
def main(
    prompt_file: Annotated[Path, typer.Argument(help="Path to prompts toml file.")],
    name1: Annotated[str, typer.Argument(help="Name of the first prompt.")],
    name2: Annotated[str, typer.Argument(help="Name of the second prompt.")],
    side_by_side: Annotated[
        bool, typer.Option("--side-by-side", "-s", help="Show side-by-side diff")
    ] = False,
) -> None:
    """Diff between two prompt texts (`gpt.prompts.PromptTemplate`)."""
    prompts: list[dict[str, str]] = tomllib.loads(prompt_file.read_text())["prompts"]

    prompt1 = _get_prompt(prompts, name1)
    prompt2 = _get_prompt(prompts, name2)

    _show_diff(prompt1, prompt2, side=side_by_side)


def _get_prompt(prompts: list[dict[str, str]], name: str) -> str:
    prompt_names = [p["name"] for p in prompts]
    prompt = next((p["prompt"] for p in prompts if p["name"] == name), None)

    if prompt is None:
        sys.exit(f"Unknown prompt named '{name}'. Choose from: {prompt_names}")
    return prompt


def _show_diff(str1: str, str2: str, *, side: bool) -> None:
    with (
        tempfile.NamedTemporaryFile(encoding="utf-8", mode="w") as f1,
        tempfile.NamedTemporaryFile(encoding="utf-8", mode="w") as f2,
    ):
        f1.write(str1)
        f1.flush()

        f2.write(str2)
        f2.flush()

        args = ["delta", f1.name, f2.name]
        if side:
            args.append("--side-by-side")
        subprocess.run(args, check=False)


if __name__ == "__main__":
    app()
