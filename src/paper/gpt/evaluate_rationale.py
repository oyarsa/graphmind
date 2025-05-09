"""Use LLM-as-judge to automatically evaluate generated novelty assessment rationales.

The input --type is one of:

- `raw`: original dataset type, `peerread.Paper`
- `graph`: output of `gpt.evaluate_paper_graph`, `PromptResult[GraphResult]`
- `paper`: output of `gpt.evaluate_paper_scimon`, `PromptResult[PaperResult]`
- `summ`: output of `gpt.summarise_related_peter.py`, `PromptResult[PaperWithRelatedSummary]`
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import statistics
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Self

import dotenv
import rich
import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table
from tqdm import tqdm

from paper import peerread as pr
from paper.gpt.evaluate_paper import EvaluationInput, PaperResult
from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import PaperWithRelatedSummary, Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    LLMClient,
    OpenAIClient,
    append_intermediate_result,
    init_remaining_items,
)
from paper.util import (
    Timer,
    cli,
    ensure_envvar,
    get_params,
    progress,
    render_params,
    sample,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)


RATIONALE_EVAL_PROMPTS = load_prompts("evaluate_rationale")


class MetricStats(BaseModel):
    """Statistics for a single metric."""

    model_config = ConfigDict(frozen=True)

    mean: float
    stdev: float


class AggregateMetrics(BaseModel):
    """Aggregate statistics for rationale metrics."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, MetricStats]


class RationaleMetrics(BaseModel):
    """Metrics from rationale evaluation."""

    metrics: Mapping[str, int]
    explanation: str

    def is_valid(self) -> bool:
        """Check if the rationale explanation is valid."""
        return self.explanation != "<e>"

    def invalid_metrics(self) -> list[str]:
        """Get names of metrics with invalid values (outside of 1-5)."""
        return [name for name, val in self.metrics.items() if val not in range(1, 6)]


class GPTRationaleEval(BaseModel):
    """Evaluation metrics from LLM judging of generated rationale."""

    model_config = ConfigDict(frozen=True)

    clarity: Annotated[
        int, Field(description="How well-written the text is. Score from 1 to 5.")
    ]
    faithfulness: Annotated[
        int,
        Field(
            description="How the rationale justifies the novety rating. Score from 1"
            " to 5."
        ),
    ]
    factuality: Annotated[
        int,
        Field(
            description="Is the rationale grounded correctly scientific facts from"
            " the main and related papers? Score from 1 to 5."
        ),
    ]
    specificity: Annotated[
        int,
        Field(
            description="Does the rationale cover information specific to the paper,"
            " or does it make overly generic statements? the main and related papers?"
            " Score from 1 to 5."
        ),
    ]
    contributions: Annotated[
        int,
        Field(
            description="Does the rationale effectively compare the main paper with the"
            " prior work? Score from 1 to 5."
        ),
    ]
    explanation: Annotated[str, Field(description="Explanation for your scores.")]

    def metrics(self) -> RationaleMetrics:
        """Get rationale metrics as a dictionary of values and an explanation."""
        return RationaleMetrics(
            metrics={
                "clarity": self.clarity,
                "faithfulness": self.faithfulness,
                "factuality": self.factuality,
                "specificity": self.specificity,
                "contributions": self.contributions,
            },
            explanation=self.explanation,
        )

    @classmethod
    def empty(cls) -> Self:
        """Empty instance of the output in case of errors."""
        return cls(
            clarity=1,
            faithfulness=1,
            factuality=1,
            specificity=1,
            contributions=1,
            explanation="<e>",
        )

    def is_valid(self) -> bool:
        """Check if this instance is invalid (created from `empty()`)."""
        return self.explanation != "<e>"


class GraphWithEval(GraphResult):
    """`GraphResult` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: RationaleMetrics

    @classmethod
    def from_(cls, graph: GraphResult, eval: RationaleMetrics) -> Self:
        """Create `GraphWithEval` from existing `GraphResult` and evaluation result."""
        return cls(graph=graph.graph, paper=graph.paper, eval_metrics=eval)


class PaperWithEval(PaperResult):
    """`PaperResult` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: RationaleMetrics

    @classmethod
    def from_(cls, paper: PaperResult, eval: RationaleMetrics) -> Self:
        """Create `PaperWithEval` from existing `PaperResult` and evaluation result."""
        return cls.model_validate({**paper.model_dump(), "eval_metrics": eval})


class PaperRawWithEval(pr.Paper):
    """`Paper` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: RationaleMetrics

    @classmethod
    def from_(cls, paper: pr.Paper, eval: RationaleMetrics) -> Self:
        """Create `PaperRawWithEval` from existing `Paper` and evaluation result."""
        return cls.model_validate({**paper.model_dump(), "eval_metrics": eval})


class PaperSummarisedWithEval(PaperWithRelatedSummary):
    """`Paper` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: RationaleMetrics

    @classmethod
    def from_(cls, paper: PaperWithRelatedSummary, eval: RationaleMetrics) -> Self:
        """Create `SummarisedPaperRawWithEval` from existing `paper` and evaluation."""
        return cls.model_validate({**paper.model_dump(), "eval_metrics": eval})


# Type for results with evaluation metrics added
type EvaluatedResult = (
    GraphWithEval | PaperWithEval | PaperRawWithEval | PaperSummarisedWithEval
)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def run(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help="The path to the JSON file containing the output of evaluation."
            " (gpt.evaluate_paper_graph, gpt.evaluate_paper_scimon or raw pr.Paper)",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="The path to the output directory where the files will be saved.",
        ),
    ],
    input_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="The type of input file",
            click_type=cli.Choice(["graph", "paper", "raw", "summ"]),
        ),
    ] = "graph",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="The model to use for evaluation."),
    ] = "gpt-4o-mini",
    limit_papers: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="The number of papers to process. Use 0 for all papers.",
        ),
    ] = 5,
    prompt: Annotated[
        str,
        typer.Option(
            help="The prompts to use for rationale evaluation.",
            click_type=cli.Choice(RATIONALE_EVAL_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results."),
    ] = False,
    keep_intermediate: Annotated[
        bool, typer.Option("--keep", help="Keep intermediate results.")
    ] = False,
    seed: Annotated[
        int,
        typer.Option(help="Random seed used for the GPT API and to shuffle the data."),
    ] = 0,
    batch_size: Annotated[
        int, typer.Option(help="Size of the batches being evaluated.")
    ] = 100,
) -> None:
    """Evaluate each paper's predicted rationale from graph or paper evaluation."""
    asyncio.run(
        evaluate_rationales(
            model,
            input_path,
            input_type,
            limit_papers,
            prompt,
            output_dir,
            continue_papers,
            continue_,
            keep_intermediate,
            seed,
            batch_size,
        )
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


async def evaluate_rationales(
    model: str,
    input_path: Path,
    input_type: str,
    limit_papers: int | None,
    prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    keep_intermediate: bool,
    seed: int,
    batch_size: int,
) -> None:
    """Evaluate each paper's predicted rationale from evaluation with LLM-as-judge.

    Args:
        model: GPT model code to use.
        input_path: Path to the JSON file containing the output of evaluation.
        input_type: Type of input file.
        limit_papers: Number of papers to process. If 0 or None, process all.
        prompt_key: Key to the prompt to use for rationale evaluation. See
            `RATIONALE_EVAL_USER_PROMPTS` for available options or `list_prompts` for
            more.
        output_dir: Directory to save the output files.
        continue_papers_file: If provided, check for entries in the input data.
        continue_: If True, use data from `continue_papers_file`.
        keep_intermediate: Keep intermediate results to be used with `continue`.
        seed: Seed for the OpenAI API call and to shuffle the data.
        batch_size: Number of items per batch.
    """
    random.seed(seed)
    params = get_params()
    logger.info(render_params(params))

    dotenv.load_dotenv()

    if limit_papers == 0:
        limit_papers = None

    client = OpenAIClient(
        api_key=ensure_envvar("OPENAI_API_KEY"), model=model, seed=seed
    )

    # TODO: Clean this up. See the other TODO in this file. (2025-05-06)
    match input_type.lower():
        case "graph":
            papers = sample(
                PromptResult.unwrap(load_data(input_path, PromptResult[GraphResult])),
                limit_papers,
            )
            result_class = GraphWithEval
        case "paper":
            papers = sample(
                PromptResult.unwrap(load_data(input_path, PromptResult[PaperResult])),
                limit_papers,
            )
            result_class = PaperWithEval
        case "raw":
            papers = sample(load_data(input_path, pr.Paper), limit_papers)
            result_class = PaperRawWithEval
        case "summ":
            papers = sample(
                PromptResult.unwrap(
                    load_data(input_path, PromptResult[PaperWithRelatedSummary])
                ),
                limit_papers,
            )
            result_class = PaperSummarisedWithEval
        case _:
            raise ValueError(
                f"Invalid input_type: {input_type}. Must be 'graph', 'paper', 'summ',"
                " or 'raw'."
            )

    prompt = RATIONALE_EVAL_PROMPTS[prompt_key]
    if not prompt.system:
        raise ValueError("Chosen prompt doesn't contain system prompt.")

    output_intermediate_file, papers_remaining = init_remaining_items(
        result_class, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _evaluate_rationales(
            client,
            prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            keep_intermediate,
            batch_size,
            result_class,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    logger.info("%s", _display_label_dist(results_items))

    metrics = calculate_aggregate_metrics(results_items)
    logger.info(">>> Results\n%s", _display_aggregate_metrics(metrics.metrics))

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)
    save_data(output_dir / "metrics.json", metrics)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


def calculate_aggregate_metrics(
    item_evals: Iterable[EvaluatedResult],
) -> AggregateMetrics:
    """Calculate mean and standard deviation for each metric."""
    item_metrics = [item.eval_metrics.metrics for item in item_evals]

    metrics_stats: dict[str, MetricStats] = {}

    for metric in sorted(item_metrics[0]):
        values = [i[metric] for i in item_metrics]
        mean_value = statistics.mean(values)
        stdev_value = statistics.stdev(values)
        metrics_stats[metric] = MetricStats(mean=mean_value, stdev=stdev_value)

    return AggregateMetrics(metrics=metrics_stats)


def _display_aggregate_metrics(metrics: dict[str, MetricStats]) -> str:
    longest_metric = max(map(len, metrics))
    return "\n".join(
        f"{name:>{longest_metric}}: mean={stats.mean:.4f} stdev={stats.stdev:.4f}"
        for name, stats in metrics.items()
    )


def _display_label_dist(item_evals: Iterable[EvaluatedResult]) -> str:
    """Display distribution of values for the metrics from evaluated items."""
    item_metrics = [item.eval_metrics.metrics for item in item_evals]
    output = [">>> Metrics distributions:"]

    for metric in sorted(item_metrics[0]):
        values = [i[metric] for i in item_metrics]
        dist = Counter(values)

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        output.append(
            f"> {metric} distribution ({mean:.4f} +- {stdev:.4f}):\n"
            + "\n".join(f"- {label}: {count}" for label, count in sorted(dist.items()))
        )

    return "\n\n".join(output)


async def _evaluate_rationales(
    client: LLMClient,
    prompt: PromptTemplate,
    items: Sequence[EvaluationInput],
    output_intermediate_file: Path,
    keep_intermediate: bool,
    batch_size: int,
    result_class: type[EvaluatedResult],
) -> GPTResult[list[PromptResult[EvaluatedResult]]]:
    """Evaluate the predicted paper rationales.

    Args:
        client: LLM client.
        prompt: User and system prompt to use for rationale evaluation.
        items: Outputs from evaluation (either GraphResult or PaperResult).
        output_intermediate_file: File to write new results after each task.
        keep_intermediate: Keep intermediate results to be used in future runs.
        batch_size: Number of items per batch.
        result_class: Class to use for the result (GraphWithEval or PaperWithEval).

    Returns:
        List of papers with evaluated rationales wrapped in a GPTResult.
    """
    results: list[PromptResult[EvaluatedResult]] = []
    total_cost = 0

    batches = list(itertools.batched(items, batch_size))
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches"), 1):
        batch_tasks = [
            _evaluate_rationale(client, item, prompt, result_class) for item in batch
        ]

        for task in progress.as_completed(
            batch_tasks, desc=f"Evaluating batch {batch_idx}"
        ):
            result = await task
            total_cost += result.cost

            results.append(result.result)
            if keep_intermediate:
                append_intermediate_result(output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


async def _evaluate_rationale(
    client: LLMClient,
    item: EvaluationInput,
    prompt: PromptTemplate,
    result_class: type[EvaluatedResult],
) -> GPTResult[PromptResult[EvaluatedResult]]:
    """Evaluate predicted rationale from evaluation.

    Args:
        client: LLM client.
        item: Output from evaluation.
        prompt: User and system prompt for rationale evaluation.
        result_class: Class to use for the result.

    Returns:
        Paper with evaluated rationale wrapped in a GPTResult.
    """
    # TODO: Clean this up. See below. (2025-05-06)
    if isinstance(item, GraphResult):
        paper = item.paper
        title = item.paper.title
        rationale_pred = paper.rationale_pred
    elif isinstance(item, PaperResult):
        paper = item
        title = item.title
        rationale_pred = item.rationale_pred
    elif isinstance(item, PaperWithRelatedSummary):
        paper = item
        title = item.paper.title
        rationale_pred = item.paper.paper.rationale
    else:
        # Must be Paper since we're using InputType
        paper = item
        title = item.title
        rationale_pred = item.rationale

    user_prompt_text = format_template(paper, rationale_pred, prompt)
    result = await client.run(GPTRationaleEval, prompt.system, user_prompt_text)
    rationale_eval = result.result or GPTRationaleEval.empty()

    if not rationale_eval.is_valid():
        logger.warning(f"Paper: '{title}': invalid rationale evaluation")

    metrics = rationale_eval.metrics()

    # TODO: clean this up (2025-03-27)
    if isinstance(item, GraphResult) and result_class == GraphWithEval:
        eval_result = GraphWithEval.from_(item, metrics)
    elif isinstance(item, PaperResult) and result_class == PaperWithEval:
        eval_result = PaperWithEval.from_(item, metrics)
    elif isinstance(item, pr.Paper) and result_class == PaperRawWithEval:
        eval_result = PaperRawWithEval.from_(item, metrics)
    elif (
        isinstance(item, PaperWithRelatedSummary)
        and result_class == PaperSummarisedWithEval
    ):
        eval_result = PaperSummarisedWithEval.from_(item, metrics)
    else:
        raise TypeError(
            f"Mismatched item type {type(item)} and result class {result_class}"
        )

    if invalid := eval_result.eval_metrics.invalid_metrics():
        logger.warning(f"{title}: invalid metric values: {invalid}")

    return GPTResult(
        result=PromptResult(
            item=eval_result,
            prompt=Prompt(system=prompt.system, user=user_prompt_text),
        ),
        cost=result.cost,
    )


def format_template(
    paper: EvaluationInput, rationale: str, prompt: PromptTemplate
) -> str:
    """Format evaluation user template using the predicted rationale."""
    if isinstance(paper, GraphResult):
        paper = paper.paper

    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        label=paper.label,
        rationale=rationale,
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts("RATIONALE EVALUATION", RATIONALE_EVAL_PROMPTS, detail=detail)


async def _evaluate_single_input(
    batch_size: int,
    input_path: Path,
    input_type: str,
    limit: int,
    model: str,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate an input file in an temporary directory and return the results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        await evaluate_rationales(
            model=model,
            input_path=input_path,
            input_type=input_type,
            limit_papers=limit,
            prompt_key=prompt,
            output_dir=output_dir,
            continue_papers_file=None,
            continue_=False,
            keep_intermediate=False,
            seed=seed,
            batch_size=batch_size,
        )

        metrics = json.loads((output_dir / "metrics.json").read_bytes())
        means = {metric: data["mean"] for metric, data in metrics["metrics"].items()}
        return {**means, "name": input_path.parent.name}


@app.command(help="Run evaluation on multiple input files", no_args_is_help=True)
def many(
    inputs: Annotated[
        list[str],
        typer.Argument(
            help="Input files to process. Each file is in the format path:type."
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for evaluation"),
    ] = "gpt-4o-mini",
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of papers to process"),
    ] = 5,
    prompt: Annotated[
        str,
        typer.Option(help="Prompt to use for evaluation"),
    ] = "simple",
    seed: Annotated[
        int,
        typer.Option(help="Random seed used for the GPT API and to shuffle the data."),
    ] = 0,
    batch_size: Annotated[
        int, typer.Option(help="Size of the batches being evaluated.")
    ] = 100,
) -> None:
    """Run evaluation on multiple input files and display metrics in a table."""
    asyncio.run(
        _evaluate_many_inputs(
            inputs=inputs,
            model=model,
            limit=limit,
            prompt=prompt,
            seed=seed,
            batch_size=batch_size,
        )
    )


async def _evaluate_many_inputs(
    inputs: list[str],
    model: str,
    limit: int,
    prompt: str,
    seed: int,
    batch_size: int,
) -> None:
    results: list[dict[str, Any]] = []

    for input_ in inputs:
        logger.warning(f"Processing: {input_}")
        try:
            if ":" in input_:
                input_path, input_type = input_.split(":", maxsplit=1)
            else:
                input_path, input_type = input_, "graph"

            results.append(
                await _evaluate_single_input(
                    batch_size, Path(input_path), input_type, limit, model, prompt, seed
                )
            )
        except Exception:
            logger.exception(f"Error processing {input_}")

    if not results:
        logger.warning("No results collected.")
        return

    _display_results(results)


def _display_results(results: list[dict[str, Any]]) -> None:
    """Display the results of evaluating multiple inputs.

    Args:
        results: List of dictionaries containing the results.
    """
    table = Table(title="Rationale Evaluation Metrics")
    table.add_column("Name", style="cyan")

    # Get all metric names (excluding 'name')
    metric_names = sorted({
        key for result in results for key in result if key != "name"
    })

    for metric in metric_names:
        table.add_column(metric.capitalize(), style="green")

    for result in sorted(results, key=lambda x: x["specificity"], reverse=True):
        table.add_row(*[
            result["name"],
            *(f"{result.get(metric, 0):.2f}" for metric in metric_names),
        ])

    rich.print(table)


if __name__ == "__main__":
    app()
