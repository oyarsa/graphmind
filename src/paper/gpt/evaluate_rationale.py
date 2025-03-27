"""Use LLM-as-judge to automatically evaluate generated novelty assessment rationales.

The input is the the output of `gpt.evaluate_paper_graph`, `PromptResult[GraphResult]`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Self

import dotenv
import typer
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from paper.gpt.evaluate_paper import PaperResult
from paper.gpt.extract_graph import GraphResult
from paper.gpt.model import Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    GPTResult,
    ModelClient,
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
        return self.explanation != "<error>"

    def invalid_metrics(self) -> list[str]:
        """Get names of metrics with invalid values (outside of 1-5)."""
        return [name for name, val in self.metrics.items() if val not in range(1, 6)]


class GPTRationaleEval(BaseModel):
    """Evaluation metrics from LLM judging of generated rationale."""

    model_config = ConfigDict(frozen=True)

    fluency: Annotated[
        int, Field(description="How well-written the text is. Score from 1 to 5.")
    ]
    faithfulness: Annotated[
        int,
        Field(
            description="How the rationale justifies the novety rating. Score from 1"
            " to 5."
        ),
    ]
    logical: Annotated[
        int,
        Field(
            description="How well are the claims supported by the evidence? Score from"
            " 1 to 5."
        ),
    ]
    explanation: Annotated[str, Field(description="Explanation for your scores.")]

    def metrics(self) -> RationaleMetrics:
        """Get rationale metrics as a dictionary of values and an explanation."""
        return RationaleMetrics(
            metrics={
                "fluency": self.fluency,
                "faithfulness": self.faithfulness,
                "logical": self.logical,
            },
            explanation=self.explanation,
        )

    @classmethod
    def empty(cls) -> Self:
        """Empty instance of the output in case of errors."""
        return cls(fluency=1, faithfulness=1, logical=1, explanation="<error>")

    def is_valid(self) -> bool:
        """Check if this instance is invalid (created from `empty()`)."""
        return self.explanation != "<error>"


class GraphWithEval(GraphResult):
    """`GraphResult` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: RationaleMetrics

    @classmethod
    def from_(cls, graph: GraphResult, eval: RationaleMetrics) -> Self:
        """Create `GraphWithEval` from existing `GraphResult` and evaluation result."""
        return cls(graph=graph.graph, paper=graph.paper, eval_metrics=eval)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    graph_path: Annotated[
        Path,
        typer.Option(
            "--graphs",
            help="The path to the JSON file containing the output of graph evaluation."
            " (gpt.evaluate_paper_graph)",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="The path to the output directory where the files will be saved.",
        ),
    ],
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
        bool, typer.Option(help="Keep intermediate results.")
    ] = False,
    seed: Annotated[
        int,
        typer.Option(help="Random seed used for the GPT API and to shuffle the data."),
    ] = 0,
    batch_size: Annotated[
        int, typer.Option(help="Size of the batches being evaluated.")
    ] = 100,
) -> None:
    """Evaluate each paper's predicted rationale from graph evaluation."""
    asyncio.run(
        evaluate_rationales(
            model,
            graph_path,
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
    graph_path: Path,
    limit_papers: int | None,
    prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    keep_intermediate: bool,
    seed: int,
    batch_size: int,
) -> None:
    """Evaluate each paper's predicted rationale from graph evaluation with LLM-as-judge.

    Args:
        model: GPT model code to use.
        graph_path: Path to the JSON file containing the output of graph evaluation.
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

    client = ModelClient(
        api_key=ensure_envvar("OPENAI_API_KEY"), model=model, seed=seed
    )

    papers = sample(
        PromptResult.unwrap(load_data(graph_path, PromptResult[GraphResult])),
        limit_papers,
    )

    prompt = RATIONALE_EVAL_PROMPTS[prompt_key]
    if not prompt.system:
        raise ValueError("Chosen prompt doesn't contain system prompt.")

    output_intermediate_file, papers_remaining = init_remaining_items(
        GraphWithEval, output_dir, continue_papers_file, papers, continue_
    )

    with Timer() as timer:
        results = await _evaluate_rationales(
            client,
            prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            keep_intermediate,
            batch_size,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    logger.info("%s", _display_label_dist(results_items))

    metrics = calculate_aggregate_metrics(results_items)
    for metric_name, stats in metrics.metrics.items():
        logger.info("%s: mean=%.4f, stdev=%.4f", metric_name, stats.mean, stats.stdev)

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)
    save_data(output_dir / "metrics.json", metrics)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


def calculate_aggregate_metrics(
    graph_evals: Iterable[GraphWithEval],
) -> AggregateMetrics:
    """Calculate mean and standard deviation for each metric."""
    graph_metrics = [graph.eval_metrics.metrics for graph in graph_evals]

    metrics_stats: dict[str, MetricStats] = {}

    for metric in sorted(graph_metrics[0]):
        values = [g[metric] for g in graph_metrics]
        mean_value = statistics.mean(values)
        stdev_value = statistics.stdev(values)
        metrics_stats[metric] = MetricStats(mean=mean_value, stdev=stdev_value)

    return AggregateMetrics(metrics=metrics_stats)


def _display_label_dist(graph_evals: Iterable[GraphWithEval]) -> str:
    """Display distribution of values for the metrics from `graph_evals`."""
    graph_metrics = [graph.eval_metrics.metrics for graph in graph_evals]
    output = [">>> Metrics distributions:"]

    for metric in sorted(graph_metrics[0]):
        values = [g[metric] for g in graph_metrics]
        dist = Counter(values)

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        output.append(
            f"> {metric} distribution ({mean:.4f} +- {stdev:.4f}):\n"
            + "\n".join(f"- {label}: {count}" for label, count in sorted(dist.items()))
        )

    return "\n\n".join(output)


async def _evaluate_rationales(
    client: ModelClient,
    prompt: PromptTemplate,
    graphs: Sequence[GraphResult],
    output_intermediate_file: Path,
    keep_intermediate: bool,
    batch_size: int,
) -> GPTResult[list[PromptResult[GraphWithEval]]]:
    """Evaluate the predicted paper rationales.

    Args:
        client: OpenAI client to use GPT.
        prompt: User and system prompt to use for rationale evaluation.
        graphs: Outputs from graph evaluation.
        output_intermediate_file: File to write new results after each task.
        keep_intermediate: Keep intermediate results to be used in future runs.
        batch_size: Number of items per batch.

    Returns:
        List of papers with evaluated rationales wrapped in a GPTResult.
    """
    results: list[PromptResult[GraphWithEval]] = []
    total_cost = 0

    batches = list(itertools.batched(graphs, batch_size))
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches"), 1):
        batch_tasks = [_evaluate_rationale(client, graph, prompt) for graph in batch]

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
    client: ModelClient, graph: GraphResult, prompt: PromptTemplate
) -> GPTResult[PromptResult[GraphWithEval]]:
    """Evaluate predicted rationale from graph evaluation.

    Args:
        client: OpenAI client to use GPT.
        graph: Output from graph evaluation.
        prompt: User and system prompt for rationale evaluation.

    Returns:
        Paper with evaluated rationale wrapped in a GPTResult.
    """

    user_prompt_text = format_template(graph.paper, graph.paper.rationale_pred, prompt)
    result = await client.run(GPTRationaleEval, prompt.system, user_prompt_text)
    rationale_eval = result.result or GPTRationaleEval.empty()

    if not rationale_eval.is_valid():
        logger.warning(f"Paper: '{graph.paper.title}': invalid rationale evaluation")

    graph_eval = GraphWithEval.from_(graph, rationale_eval.metrics())

    if invalid := graph_eval.eval_metrics.invalid_metrics():
        logger.warning(f"{graph.paper.title}: invalid metric values: {invalid}")

    return GPTResult(
        result=PromptResult(
            item=graph_eval,
            prompt=Prompt(system=prompt.system, user=user_prompt_text),
        ),
        cost=result.cost,
    )


def format_template(paper: PaperResult, rationale: str, prompt: PromptTemplate) -> str:
    """Format evaluation user template using the predicted rationale."""
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


if __name__ == "__main__":
    app()
