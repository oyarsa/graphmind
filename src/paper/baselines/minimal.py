"""Minimal LLM baselines: perform evaluation on OpenRouter models as plain text."""

# pyright: basic

import asyncio
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from paper import peerread as pr
from paper.evaluation_metrics import Metrics, calculate_paper_metrics
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    EVALUATE_DEMONSTRATIONS,
    get_demonstrations,
)
from paper.gpt.run_gpt import GPTResult, ModelClient
from paper.util import cli, ensure_envvar, progress, sample
from paper.util.serde import load_data, save_data

SYSTEM_PROMPT = (
    """Given the following scientific paper, determine whether it's novel."""
)
USER_PROMPT = """
The following data contains information about a scientific paper. It includes the \
paper's title and abstract.

Based on this content, decide whether the paper is novel enough or not. If it is, give \
it a label of 1. If it isn't, give it a label of 0. This should reflect how much the \
paper brings and develops new ideas previously unseen in the literature. First, generate \
the rationale for your novelty rating, then give the final novelty rating.

If the paper approval decision is "True", the novelty label should be 1 (novel).

The output should have the following format:

```
Rationale: <text>

Label: <0 or 1>
```

#####
{demonstrations}

-Data-
Title: {title}
Abstract: {abstract}
Approval decision: {approval}

#####
Output:
"""


class Result(BaseModel):
    """Result of paper evaluation."""

    model_config = ConfigDict(frozen=True)

    y_true: int
    y_pred: int
    prompt: str
    output: str | None


class ModelResult(BaseModel):
    """Results for a specific model."""

    model_config = ConfigDict(frozen=True)

    name: str
    metrics: Metrics
    cost: float
    results: list[Result]


async def evaluate_paper(
    client: ModelClient, paper: pr.Paper, demonstrations: str
) -> GPTResult[Result]:
    """Evaluate `paper` using the given `client`."""
    user_text = USER_PROMPT.format(
        title=paper.title,
        abstract=paper.abstract,
        approval=paper.approval,
        demonstrations=demonstrations,
    )

    result = await client.plain(SYSTEM_PROMPT, user_text)
    predicted = parse_result(result.result) if result.result else 0

    return GPTResult(
        result=Result(
            y_true=paper.label, y_pred=predicted, prompt=user_text, output=result.result
        ),
        cost=result.cost,
    )


def parse_result(text_: str) -> int:
    """Parse the output text to get the label.

    The output should have the following format:

        Label: <0 or 1>
    """
    text = text_.casefold()
    for line_ in text.splitlines():
        line = line_.strip()
        if not line.startswith("label:"):
            continue

        rest = line.removeprefix("label:").strip()
        try:
            return int(rest[0])
        except Exception:
            pass

    raise ValueError(f"Invalid output: {text_}")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Path to JSON file with input data.")
    ],
    models: Annotated[
        list[str],
        typer.Option(
            "--model", "-m", help="OpenRouter models to use (can specify multiple)."
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Path to output JSON file with predictions"
        ),
    ],
    num_papers: Annotated[
        int | None, typer.Option("--num-papers", "-n", help="Number of papers to query")
    ] = None,
    chart_file: Annotated[
        Path | None,
        typer.Option(
            "--chart",
            "-c",
            help="Path to save the chart (PDF). If not provided, displays the chart"
            " graphically.",
        ),
    ] = None,
    demos: Annotated[
        str | None,
        typer.Option(
            help="Name of file containing demonstrations to use in few-shot prompt.",
            click_type=cli.Choice(EVALUATE_DEMONSTRATIONS),
        ),
    ] = "orc_10",
    demo_prompt: Annotated[
        str,
        typer.Option(
            help="User prompt to use for building the few-shot demonstrations.",
            click_type=cli.Choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ] = "abstract",
) -> None:
    """Evaluate papers with OpenRouter models."""
    asyncio.run(
        async_main(
            input_file, models, output_file, num_papers, chart_file, demos, demo_prompt
        )
    )


async def async_main(
    input_file: Path,
    models: list[str],
    output_file: Path,
    num_papers: int | None,
    chart_file: Path | None,
    demonstrations_key: str | None,
    demo_prompt_key: str,
) -> None:
    """Evaluate papers in `input_file` using each model in `models`."""
    papers = sample(load_data(input_file, pr.Paper), num_papers)
    demonstrations = get_demonstrations(demonstrations_key, demo_prompt_key)

    model_results: list[ModelResult] = []
    for model in models:
        result = await evaluate_model(model, papers, demonstrations)
        model_results.append(result)

    display_results_table(model_results)
    generate_bar_chart(model_results, chart_file)
    save_data(output_file, model_results)


def display_results_table(model_results: list[ModelResult]) -> None:
    """Display a table with the results for each model.

    Args:
        model_results: List of model results.
    """
    console = Console()
    table = Table(title="Model Evaluation Results")

    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1 Score", style="green")
    table.add_column("Cost", style="yellow")

    for result in model_results:
        metrics = result.metrics
        table.add_row(
            result.name,
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1:.4f}",
            f"${result.cost:.4f}",
        )

    console.print(table)


async def evaluate_model(
    model: str, papers: list[pr.Paper], demonstrations: str
) -> ModelResult:
    """Evaluate all papers using the specified model."""
    client = ModelClient(
        api_key=ensure_envvar("OPENROUTER_API_KEY"),
        seed=0,
        base_url="https://openrouter.ai/api/v1",
        model=model,
    )

    tasks = [evaluate_paper(client, paper, demonstrations) for paper in papers]
    results: list[Result] = []
    total_cost = 0

    for task in progress.as_completed(tasks, desc=model):
        result = await task
        results.append(result.result)
        total_cost += result.cost

    return ModelResult(
        name=model,
        metrics=calculate_paper_metrics(results, total_cost),
        cost=total_cost,
        results=results,
    )


def generate_bar_chart(
    model_results: list[ModelResult], output_path: Path | None = None
) -> None:
    """Generate a bar chart comparing the accuracy of different models.

    Args:
        model_results: List of model results.
        output_path: Optional path to save the chart (PDF). If None, displays the chart
            graphically.
    """
    models = [result.name for result in model_results]
    accuracies = [result.metrics.accuracy for result in model_results]
    f1_scores = [result.metrics.f1 for result in model_results]

    # Create figure with two bar sets (one for accuracy, one for F1)
    _, ax = plt.subplots(figsize=(10, 6))

    # Set width of bars
    bar_width = 0.35
    x = range(len(models))

    # Create bars
    ax.bar(
        [i - bar_width / 2 for i in x],
        accuracies,
        bar_width,
        label="Accuracy",
        color="steelblue",
    )
    ax.bar(
        [i + bar_width / 2 for i in x],
        f1_scores,
        bar_width,
        label="F1 Score",
        color="lightcoral",
    )

    # Add labels and title
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    if output_path:
        print(f"Bar chart saved to {output_path}")
        plt.savefig(output_path)
        plt.close()
    else:
        # Display interactively
        print("Chart displayed interactively")
        plt.show()


if __name__ == "__main__":
    app()
