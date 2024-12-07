"""Run experiments with citation context classification across multiple combinations.

Runs combinations of:
- prompts: simple, full, sentence
- models: GPT-4o, GPT-4o-mini
"""

import itertools
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from paper.util import Timer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path, typer.Argument(help="Path to JSON input file to annotate.")
    ],
    base_output: Annotated[
        Path, typer.Argument(help="Base output directory: one subdir per combination.")
    ],
    ref_limit: Annotated[
        int,
        typer.Option(help="Number of references to run. If 0, run everything."),
    ] = 0,
    paper_limit: Annotated[
        int,
        typer.Option(help="Number of papers to run. If 0, run everything."),
    ] = 0,
    clean: Annotated[
        bool, typer.Option(help="Clean existing results before running experiments.")
    ] = False,
) -> None:
    """Run multiple classification experiments across configuration combinations."""
    print(f"Output directory: {base_output}")

    if clean:
        print("Removing existing experiment results...")
        shutil.rmtree(base_output, ignore_errors=True)
    else:
        print("Skipping removal of existing experiment results.")

    print(f"Running on {"'all'" if ref_limit == 0 else ref_limit} references.")
    print()

    prompts = ["full", "simple", "sentence"]
    models = ["gpt-4o-mini", "gpt-4o"]

    combinations = list(itertools.product(prompts, models))

    timer = Timer()
    timer.start()

    for i, (prompt, model) in enumerate(combinations, start=1):
        name = f"{prompt}_{model}"
        output_dir = base_output / name

        cmd = [
            "uv",
            "run",
            "gpt",
            "context",
            "run",
            input_file,
            output_dir,
            "--user-prompt",
            prompt,
            "--model",
            model,
            "--ref-limit",
            ref_limit,
            "--limit",
            paper_limit,
        ]

        print(f"\033[33m[{i}/{len(combinations)}] Running with: {name}\033[0m")
        subprocess.run(_strs(*cmd), check=False)
        print()

    script_dir = Path(__file__).parent
    explore_script = script_dir / "explore_context_experiments.py"
    subprocess.run(_strs("uv", "run", explore_script, base_output), check=False)

    timer.stop()
    print(f"\nTotal time taken: {timer.human}")


def _strs(*args: object) -> list[str]:
    return [str(x) for x in args]


if __name__ == "__main__":
    app()
