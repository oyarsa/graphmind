"""Extract sample rationales from experiment results.

Outputs rationales in Markdown format, separated by predicted rating ranges (1-2 vs 3+).
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import PromptResult
from paper.util.serde import load_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


def format_rationale(item: PromptResult[GraphResult]) -> str:
    """Format a single paper's rationale as valid Markdown."""
    paper = item.item.paper

    return f"""---

### {paper.title}

**True Rating:** {paper.originality_rating}
**Predicted Rating:** {paper.y_pred}

**Rationale:**

{paper.rationale_pred}

"""


def extract_rationales(input_file: Path, sample_size: int = 5) -> None:
    """Extract sample rationales from experiment results."""
    # Load data with proper types
    data = load_data(input_file, PromptResult[GraphResult])

    # Separate by prediction range
    low_predictions = [item for item in data if item.item.paper.y_pred <= 2]
    high_predictions = [item for item in data if item.item.paper.y_pred >= 3]

    # Take samples
    low_sample = low_predictions[:sample_size]
    high_sample = high_predictions[:sample_size]

    # Generate output
    output_lines = [
        "# Rationales by Predicted Rating - Sample",
        "",
        f"## Section 1: Predicted Ratings 1-2 (Low Novelty) - Sample of {sample_size}",
        "",
    ]

    for item in low_sample:
        output_lines.append(format_rationale(item))

    output_lines.extend([
        "",
        f"## Section 2: Predicted Ratings 3+ (Moderate to High Novelty) - Sample of {sample_size}",
        "",
    ])

    for item in high_sample:
        output_lines.append(format_rationale(item))

    # Print to stdout
    print("\n".join(output_lines))

    total_papers = len(low_sample) + len(high_sample)
    print(f"\nDone! Extracted {total_papers} papers", file=sys.stderr)
    print(f"  - Low predictions (1-2): {len(low_sample)}", file=sys.stderr)
    print(f"  - High predictions (3+): {len(high_sample)}", file=sys.stderr)


@app.command(help=__doc__)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to compressed JSON result file (.json.zst)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            "-n",
            help="Number of samples to extract per section",
        ),
    ] = 5,
) -> None:
    """Extract sample rationales from experiment results.

    Outputs rationales in Markdown format to stdout, separated by predicted
    rating ranges (1-2 vs 3+). Status messages are printed to stderr.

    Examples:
        python extract_rationales.py output/eval_orc/ablation_full/run_4/result.json.zst

        python extract_rationales.py input.json.zst --sample-size 10 > output.md
    """
    extract_rationales(input_file, sample_size)


if __name__ == "__main__":
    app()
