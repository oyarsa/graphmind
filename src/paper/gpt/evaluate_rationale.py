"""Use LLM-as-judge to automatically evaluate generated novelty assessment rationales.

The input is the the output of `gpt.evaluate_paper_graph`, `PromptResult[GraphResult]`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Self, override

import dotenv
import typer
from pydantic import BaseModel, ConfigDict, Field

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
    setup_logging,
    shuffled,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)


RATIONALE_EVAL_PROMPTS = load_prompts("evaluate_rationale")


class GPTRationaleMetrics(BaseModel):
    """Metrics from rationale evaluation."""

    metrics: Mapping[str, int]
    explanation: str

    def is_valid(self) -> bool:
        """Check if the rationale explanation is valid."""
        return self.explanation == "<error>"


class GPTRationaleBase(ABC, BaseModel):
    """Base class rationale evaluation metrics."""

    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        """Return empty instance. Used for error handling."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the instance is valid (not empty)."""

    @abstractmethod
    def metrics(self) -> GPTRationaleMetrics:
        """All metrics in dictionary form."""


class GPTRationaleEvalSimple(GPTRationaleBase):
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

    @override
    def metrics(self) -> GPTRationaleMetrics:
        return GPTRationaleMetrics(
            metrics={
                "fluency": self.fluency,
                "faithfulness": self.faithfulness,
                "logical": self.logical,
            },
            explanation=self.explanation,
        )

    @override
    @classmethod
    def empty(cls) -> Self:
        return cls(fluency=1, faithfulness=1, logical=1, explanation="<error>")

    @override
    def is_valid(self) -> bool:
        return self.explanation == "<error>"


class GraphWithEval(GraphResult):
    """`GraphResult` with LLM-as-judge evaluation of generated rationale."""

    eval_metrics: GPTRationaleMetrics

    @classmethod
    def from_(cls, graph: GraphResult, eval: GPTRationaleMetrics) -> Self:
        """Create `GraphWithEval` from existing `GraphResult` and evaluation result."""
        return cls(graph=graph.graph, paper=graph.paper, eval_metrics=eval)


EVAL_TYPES: Mapping[str, type[GPTRationaleBase]] = {
    "simple": GPTRationaleEvalSimple,
}

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

    papers = PromptResult.unwrap(
        shuffled(load_data(graph_path, PromptResult[GraphResult]))
    )[:limit_papers]

    prompt = RATIONALE_EVAL_PROMPTS[prompt_key]
    if not prompt.system:
        raise ValueError("Chosen prompt doesn't contain system prompt.")

    if prompt.type_name not in EVAL_TYPES:
        valid_types = sorted(EVAL_TYPES)
        raise ValueError(
            f"Prompt type '{prompt.type_name}' is invalid. Must be one of {valid_types}."
        )

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
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    logger.info("%s", _display_label_dist(results_items))

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


def _display_label_dist(graph_evals: Sequence[GraphWithEval]) -> str:
    """Display distribution of values for the metrics from `graph_evals`."""
    graph_metrics = [graph.eval_metrics.metrics for graph in graph_evals]
    output = [">>> Metrics distributions:"]

    for metric in sorted(graph_metrics[0]):
        dist = Counter(g[metric] for g in graph_metrics)
        total = sum(dist.values())

        output.append(
            f"> {metric} distribution ({total}):\n"
            + "\n".join(f"- {label}: {count}" for label, count in sorted(dist.items()))
        )

    return "\n\n".join(output)


async def _evaluate_rationales(
    client: ModelClient,
    prompt: PromptTemplate,
    graphs: Sequence[GraphResult],
    output_intermediate_file: Path,
    keep_intermediate: bool,
) -> GPTResult[list[PromptResult[GraphWithEval]]]:
    """Evaluate the predicted paper rationales.

    Args:
        client: OpenAI client to use GPT.
        prompt: User and system prompt to use for rationale evaluation.
        graphs: Outputs from graph evaluation.
        output_intermediate_file: File to write new results after each task.
        keep_intermediate: Keep intermediate results to be used in future runs.

    Returns:
        List of papers with evaluated rationales wrapped in a GPTResult.
    """
    results: list[PromptResult[GraphWithEval]] = []
    total_cost = 0

    tasks = [_evaluate_rationale(client, graph, prompt) for graph in graphs]

    for task in progress.as_completed(tasks, desc="Processing papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        if keep_intermediate:
            append_intermediate_result(
                GraphWithEval, output_intermediate_file, result.result
            )

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
    type_ = EVAL_TYPES[prompt.type_name]
    result = await client.run(type_, prompt.system, user_prompt_text)
    rationale_eval = result.result or type_.empty()

    if not rationale_eval.is_valid():
        logger.warning(f"Paper: '{graph.paper.title}': invalid rationale evaluation")

    graph_eval = GraphWithEval.from_(graph, rationale_eval.metrics())
    wrong = [
        name
        for name, val in graph_eval.eval_metrics.metrics.items()
        if val not in range(1, 6)
    ]
    if wrong:
        logger.warning(f"{graph.paper.title}: invalid metric values: {wrong}")

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
        rating=paper.y_pred,
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
