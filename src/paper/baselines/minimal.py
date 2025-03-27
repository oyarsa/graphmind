"""Minimal LLM baselines: perform evaluation on OpenRouter models as plain text."""

import asyncio
import dataclasses as dc
from pathlib import Path
from typing import Annotated

import typer

from paper import peerread as pr
from paper.evaluation_metrics import calculate_paper_metrics
from paper.gpt.run_gpt import GPTResult, ModelClient
from paper.util import ensure_envvar, progress, sample
from paper.util.serde import load_data

SYSTEM_PROMPT = """
Given the following scientific paper, extract the important entities from the text and \
the relationships between them. The goal is to build a faithful and concise representation \
of the paper that captures the most important elements necessary to evaluate its novelty.
"""
USER_PROMPT = """
The following data contains information about a scientific paper. It includes the \
paper's title and abstract.

Based on this content, decide whether the paper is novel enough or not. If it is, give \
it a label of 1. If it isn't, give it a label of 0. This should reflect how much the \
paper brings and develops new ideas previously unseen in the literature. First, generate \
the rationale for your novelty rating, then give the final novelty rating.

The output should have the following format:

```
Label: <0 or 1>
```

#####
-Data-
Title: {title}
Abstract: {abstract}

#####
Output:
"""


@dc.dataclass(frozen=True)
class Result:
    """Result of paper evaluation."""

    y_true: int
    y_pred: int


async def _evaluate_paper(client: ModelClient, paper: pr.Paper) -> GPTResult[Result]:
    user_text = USER_PROMPT.format(title=paper.title, abstract=paper.abstract)
    result = await client.plain(SYSTEM_PROMPT, user_text)
    predicted = parse_result(result.result) if result.result else 0
    return GPTResult(result=Result(paper.label, predicted), cost=result.cost)


def parse_result(text_: str) -> int:
    """Parse the output text to get the label.

    The output should have the following format:

        Label: <0 or 1>
    """
    text = text_.casefold()
    for line_ in text.splitlines():
        line = line_.strip()
        if not line.startswith("label"):
            continue

        rest = line.removeprefix("label:").strip()
        try:
            return int(rest[0])
        except Exception:
            pass

    raise ValueError(f"Invalid output: {text}")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input: Annotated[
        Path, typer.Option("--input", "-i", help="Path to JSON file with input data.")
    ],
    model: Annotated[str, typer.Option("--model", "-m", help="OpenRouter model used.")],
    num_papers: Annotated[
        int | None, typer.Option("--num-papers", "-n", help="Number of papers to query")
    ] = None,
) -> None:
    """Evaluate papers with OpenRouter models."""
    asyncio.run(_async_main(input, model, num_papers))


async def _async_main(file: Path, model: str, num_papers: int | None) -> None:
    papers = sample(load_data(file, pr.Paper), num_papers)

    client = ModelClient(
        api_key=ensure_envvar("OPENROUTER_API_KEY"),
        seed=0,
        base_url="https://openrouter.ai/api/v1",
        model=model,
    )

    tasks = [_evaluate_paper(client, paper) for paper in papers]
    results: list[Result] = []
    total_cost = 0

    for task in progress.as_completed(tasks):
        result = await task
        results.append(result.result)
        total_cost += result.cost

    metrics = calculate_paper_metrics(results, total_cost)
    print(metrics)


if __name__ == "__main__":
    app()
