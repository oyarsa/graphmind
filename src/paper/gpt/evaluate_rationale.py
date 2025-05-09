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
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import dotenv
import rich
import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from paper import peerread as pr
from paper.gpt.evaluate_paper import PaperResult
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
PAIRWISE_COMPARISON_PROMPTS = load_prompts("pairwise_comparison")

# Elo rating constants
DEFAULT_ELO = 1200  # Starting rating for all players
K_FACTOR = 32  # How much ratings can change in a single match
EXPECTED_SCORE_DIVISOR = 400  # For expected score calculation


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


class PairwiseComparisonResult(BaseModel):
    """Result from pairwise comparison of two rationales by LLM."""

    model_config = ConfigDict(frozen=True)

    winner: Literal["A", "B", "tie"]
    score: float
    """1.0 for clear winner, 0.5 for tie."""
    explanation: str
    metric: str
    """Which metric this comparison is for."""


# Type alias for input items
type InputType = GraphResult | PaperResult | pr.Paper | PaperWithRelatedSummary


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


@dataclass(frozen=True, kw_only=True)
class PlayerMatch:
    """Match in player match history."""

    player: str
    opponent: str
    score: float
    explanation: str


@dataclass
class EloPlayer:
    """Represents a player in the Elo rating system."""

    name: str
    rating: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    ties: int = 0
    matches_played: int = 0

    # Track all match results for detailed analysis
    match_history: list[PlayerMatch] = field(default_factory=list)

    def add_match_result(self, opponent: str, score: float, explanation: str) -> None:
        """Add a match result to the player's history.

        Args:
            opponent: Name of the opponent.
            score: 1.0 for win, 0.5 for tie, 0.0 for loss.
            explanation: Explanation for the match result.
        """
        self.match_history.append(
            PlayerMatch(
                player=self.name,
                opponent=opponent,
                score=score,
                explanation=explanation,
            )
        )
        self.matches_played += 1

        if score == 1.0:
            self.wins += 1
        elif score == 0.5:
            self.ties += 1
        else:
            self.losses += 1

    def update_rating(self, expected_score: float, actual_score: float) -> None:
        """Update the Elo rating based on match outcome.

        Args:
            expected_score: Expected probability of winning (0.0-1.0).
            actual_score: Actual outcome (1.0=win, 0.5=tie, 0.0=loss).
        """
        self.rating += K_FACTOR * (actual_score - expected_score)

    @property
    def win_percentage(self) -> float:
        """Calculate win percentage of the player."""
        if self.matches_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.matches_played


@dataclass(frozen=True, kw_only=True)
class TournamentMatch:
    """Entry in tournament match history."""

    player_a: str
    player_b: str
    result: PairwiseComparisonResult


class TournamentSystem:
    """Manages an Elo tournament for rationale comparisons."""

    def __init__(self, model_names: Sequence[str], metric: str) -> None:
        """Initialize the tournament system.

        Args:
            model_names: Names of the different models being compared.
            metric: The metric this tournament is evaluating.
        """
        self.metric = metric
        self.players: dict[str, EloPlayer] = {
            name: EloPlayer(name=name) for name in model_names
        }
        self.matches: list[TournamentMatch] = []

    def calculate_expected_score(
        self, player_a: EloPlayer, player_b: EloPlayer
    ) -> float:
        """Calculate expected score for player A against player B.

        Args:
            player_a: First player.
            player_b: Second player.

        Returns:
            Expected probability of player A winning (0.0-1.0).
        """
        return 1.0 / (
            1.0 + 10.0 ** ((player_b.rating - player_a.rating) / EXPECTED_SCORE_DIVISOR)
        )

    def record_match(
        self, player_a_name: str, player_b_name: str, result: PairwiseComparisonResult
    ) -> None:
        """Record the outcome of a match and update player ratings.

        Args:
            player_a_name: Name of the first player.
            player_b_name: Name of the second player.
            result: The result of the comparison.
        """
        # Ensure both players exist
        player_a = self.players[player_a_name]
        player_b = self.players[player_b_name]

        # Calculate expected scores
        expected_a = self.calculate_expected_score(player_a, player_b)
        expected_b = 1.0 - expected_a

        # Determine actual scores from the result
        if result.winner == "A":
            actual_a, actual_b = 1.0, 0.0
        elif result.winner == "B":
            actual_a, actual_b = 0.0, 1.0
        else:  # Tie
            actual_a, actual_b = 0.5, 0.5

        # Update ratings
        player_a.update_rating(expected_a, actual_a)
        player_b.update_rating(expected_b, actual_b)

        # Record match results for each player
        player_a.add_match_result(player_b_name, actual_a, result.explanation)
        player_b.add_match_result(player_a_name, actual_b, result.explanation)

        # Save match in tournament history
        self.matches.append(
            TournamentMatch(
                player_a=player_a_name, player_b=player_b_name, result=result
            )
        )

    def get_rankings(self) -> list[tuple[str, float, int, int, int]]:
        """Get the current rankings of players.

        Returns:
            List of tuples with (name, rating, wins, losses, ties) sorted by rating.
        """
        return sorted(
            (
                (p.name, p.rating, p.wins, p.losses, p.ties)
                for p in self.players.values()
            ),
            key=lambda x: x[1],
            reverse=True,
        )

    def get_rankings_with_rank(self) -> list[PlayerRank]:
        """Get rankings with rank position.

        Returns:
            List of PlayerRank with rank data for all players.
        """
        rankings = self.get_rankings()
        return [
            PlayerRank(
                rank=i,
                name=name,
                rating=rating,
                wins=wins,
                losses=losses,
                ties=ties,
            )
            for i, (name, rating, wins, losses, ties) in enumerate(rankings, 1)
        ]


@dataclass(frozen=True, kw_only=True)
class PlayerRank:
    """Player entry in ranking."""

    rank: int
    name: str
    rating: float
    wins: int
    losses: int
    ties: int


class TournamentManager:
    """Manages multiple tournaments for different metrics."""

    def __init__(self, model_names: Sequence[str], metrics: Sequence[str]) -> None:
        """Initialize tournament manager.

        Args:
            model_names: Names of the different models being compared.
            metrics: Metrics to run tournaments for.
        """
        self.tournaments = {
            metric: TournamentSystem(model_names, metric) for metric in metrics
        }
        self.model_names = model_names
        self.metrics = metrics

    def record_match(
        self, player_a: str, player_b: str, result: PairwiseComparisonResult
    ) -> None:
        """Record a match result in the appropriate tournament.

        Args:
            player_a: Name of the first player.
            player_b: Name of the second player.
            result: Comparison result from the LLM.
        """
        self.tournaments[result.metric].record_match(player_a, player_b, result)

    def get_overall_ranks(self) -> dict[str, dict[str, Any]]:
        """Calculate overall rankings based on average ranks across metrics.

        Returns:
            Dictionary with model names as keys and rank statistics as values.
        """
        # For each model, collect its rank in each tournament
        model_ranks: dict[str, list[int]] = {name: [] for name in self.model_names}

        for metric in self.metrics:
            tournament = self.tournaments[metric]
            rankings = tournament.get_rankings_with_rank()
            for player in rankings:
                model_ranks[player.name].append(player.rank)

        # Calculate mean and median ranks
        return {
            model: {
                "ranks": ranks,
                "mean_rank": statistics.mean(ranks),
                "median_rank": statistics.median(ranks),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "metric_ranks": {
                    metric: next(
                        player.rank
                        for player in self.tournaments[metric].get_rankings_with_rank()
                        if player.name == model
                    )
                    for metric in self.metrics
                },
            }
            for model, ranks in model_ranks.items()
        }


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
    items: Sequence[InputType],
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
    item: InputType,
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


def format_template(paper: InputType, rationale: str, prompt: PromptTemplate) -> str:
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


class PaperMetadata(BaseModel):
    """Metadata for paper being compared."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    abstract: str
    label: int
    rationale: str


def extract_metadata(paper: InputType) -> PaperMetadata:
    """Extract title and abstract and other metadata needed for prompts.

    Args:
        paper: Paper object

    Returns:
        PaperMetadata object with title, abstract, etc.
    """
    match paper:
        case GraphResult():
            return PaperMetadata(
                id=paper.id,
                title=paper.paper.title,
                abstract=paper.paper.abstract,
                label=paper.paper.label,
                rationale=paper.paper.rationale,
            )
        case PaperResult():
            return PaperMetadata(
                id=paper.id,
                title=paper.title,
                abstract=paper.abstract,
                label=paper.label,
                rationale=paper.rationale,
            )
        case PaperWithRelatedSummary():
            return PaperMetadata(
                id=paper.id,
                title=paper.paper.title,
                abstract=paper.paper.abstract,
                label=paper.label,
                rationale=paper.paper.paper.rationale,
            )
        case pr.Paper():
            # Must be pr.Paper
            return PaperMetadata(
                id=paper.id,
                title=paper.title,
                abstract=paper.abstract,
                label=paper.label,
                rationale=paper.rationale,
            )


def find_common_papers(
    paper_collections: Sequence[Sequence[InputType]],
) -> dict[str, list[InputType]]:
    """Find papers that exist in all collections based on ID.

    Args:
        paper_collections: List of lists of paper objects.

    Returns:
        Dictionary mapping paper IDs to list of paper objects from each collection.
    """
    # Extract IDs from each collection
    id_sets = [{extract_metadata(p).id for p in papers} for papers in paper_collections]

    # Find IDs common to all collections
    common_ids = set[str].intersection(*id_sets) if id_sets else set[str]()

    # Group papers by ID
    result: dict[str, list[InputType]] = {}

    for paper_id in common_ids:
        paper_group: list[InputType] = []

        for papers in paper_collections:
            for paper in papers:
                if extract_metadata(paper).id == paper_id:
                    paper_group.append(paper)
                    break

        result[paper_id] = paper_group

    return result


async def compare_rationales(
    client: LLMClient,
    paper_metadata: PaperMetadata,
    rationale_a: str,
    rationale_b: str,
    model_a: str,
    model_b: str,
    metric: str,
    prompt: PromptTemplate,
) -> GPTResult[PairwiseComparisonResult]:
    """Compare two rationales for the same paper using LLM.

    Args:
        client: LLM client.
        paper_metadata: Paper metadata (title, abstract, etc.).
        rationale_a: First rationale to compare.
        rationale_b: Second rationale to compare.
        model_a: Name of the first model.
        model_b: Name of the second model.
        metric: The metric to focus on in the comparison.
        prompt: Prompt template for the comparison.

    Returns:
        Comparison result wrapped in a GPTResult.
    """
    user_prompt_text = prompt.template.format(
        title=paper_metadata.title,
        abstract=paper_metadata.abstract,
        label=paper_metadata.label,
        rationale_a=rationale_a,
        rationale_b=rationale_b,
        model_a=model_a,
        model_b=model_b,
        metric=metric,
    )

    result = await client.run(PairwiseComparisonResult, prompt.system, user_prompt_text)
    return result.map(
        lambda r: r
        if r is not None
        else PairwiseComparisonResult(
            winner="tie",
            score=0.5,
            explanation="Evaluation error. Setting to tie.",
            metric=metric,
        )
    )


async def run_tournament(
    client: LLMClient,
    common_papers: dict[str, list[InputType]],
    model_names: list[str],
    output_dir: Path,
    tournament_prompt_key: str,
    metrics: list[str],
    seed: int,
) -> dict[str, Any]:
    """Run a pairwise Elo tournament between multiple models.

    Args:
        client: LLM client.
        common_papers: Dictionary mapping paper IDs to papers from each model.
        model_names: Names of the models being compared.
        output_dir: Directory to save results.
        model: GPT model to use.
        tournament_prompt_key: Key for the comparison prompt to use.
        metrics: Metrics to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Tournament results
    """
    random.seed(seed)

    # Initialize tournament manager
    manager = TournamentManager(model_names, metrics)

    # Get the comparison prompt
    prompt = PAIRWISE_COMPARISON_PROMPTS[tournament_prompt_key]
    if not prompt.system:
        raise ValueError("Chosen prompt doesn't contain system prompt.")

    # Create all possible pairings of models (A vs B, A vs C, B vs C, etc.)
    model_pairs = list(itertools.permutations(range(len(model_names)), 2))

    # Convert paper IDs to list for random ordering
    paper_ids = list(common_papers.keys())
    random.shuffle(paper_ids)

    total_cost = 0.0
    total_comparisons = len(paper_ids) * len(model_pairs) * len(metrics)

    with tqdm(total=total_comparisons, desc="Running Elo tournament") as pbar:
        for paper_id in paper_ids:
            papers = common_papers[paper_id]
            paper_metadata = extract_metadata(papers[0])

            for metric in metrics:
                for i, j in model_pairs:
                    model_a = model_names[i]
                    model_b = model_names[j]

                    rationale_a = extract_metadata(papers[i]).rationale
                    rationale_b = extract_metadata(papers[j]).rationale

                    comparison_result = await compare_rationales(
                        client,
                        paper_metadata,
                        rationale_a,
                        rationale_b,
                        model_a,
                        model_b,
                        metric,
                        prompt,
                    )

                    manager.record_match(model_a, model_b, comparison_result.result)

                    total_cost += comparison_result.cost
                    pbar.update(1)

    overall_ranks = manager.get_overall_ranks()

    # Save tournament results
    tournament_results = {
        "model_names": model_names,
        "metrics": metrics,
        "paper_count": len(paper_ids),
        "total_comparisons": total_comparisons,
        "total_cost": total_cost,
        "metric_rankings": {
            metric: [
                {
                    "rank": player.rank,
                    "name": player.name,
                    "rating": player.rating,
                    "wins": player.wins,
                    "losses": player.losses,
                    "ties": player.ties,
                }
                for player in tournament.get_rankings_with_rank()
            ]
            for metric, tournament in manager.tournaments.items()
        },
        "overall_rankings": [
            {
                "name": name,
                "mean_rank": stats["mean_rank"],
                "median_rank": stats["median_rank"],
                "best_rank": stats["best_rank"],
                "worst_rank": stats["worst_rank"],
                "metric_ranks": stats["metric_ranks"],
            }
            for name, stats in sorted(
                overall_ranks.items(), key=lambda x: x[1]["mean_rank"]
            )
        ],
    }

    save_data(output_dir / "tournament_results.json", tournament_results)

    return tournament_results


def _display_tournament_results(results: dict[str, Any]) -> str:
    """Format tournament results for display.

    Args:
        results: Tournament results dictionary.

    Returns:
        Formatted string for display.
    """
    table = Table(title="Elo Tournament Rankings")

    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Model", style="green")
    table.add_column("Mean Rank", justify="right")
    table.add_column("Median Rank", justify="right")

    metrics = list(results["metric_rankings"].keys())
    for metric in metrics:
        table.add_column(metric.capitalize(), justify="right")

    # Add rows for each model's overall ranking
    for i, model in enumerate(results["overall_rankings"], 1):
        name = model["name"]
        mean_rank = f"{model['mean_rank']:.2f}"
        median_rank = f"{model['median_rank']:.1f}"

        # Get metric-specific ranks
        metric_ranks = [str(model["metric_ranks"][m]) for m in metrics]

        table.add_row(str(i), name, mean_rank, median_rank, *metric_ranks)

    return _console_to_str(table)


def _console_to_str(*objects: Any) -> str:
    buf = StringIO()
    console = Console(file=buf, force_jupyter=False)
    console.print(*objects)
    return buf.getvalue()


TOURNAMENT_ALL_METRICS = [
    "clarity",
    "faithfulness",
    "factuality",
    "specificity",
    "contributions",
]


@app.command(no_args_is_help=True)
def tournament(
    inputs: Annotated[
        list[str],
        typer.Argument(
            help="Input files to process. Each file is in the format path:type:name."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="The path to the output directory where results will be saved.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for evaluation"),
    ] = "gpt-4o-mini",
    tournament_prompt: Annotated[
        str,
        typer.Option(
            help="The prompts to use for pairwise comparison.",
            click_type=cli.Choice(PAIRWISE_COMPARISON_PROMPTS),
        ),
    ] = "standard",
    metrics: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            help="Metrics to evaluate in tournament",
            click_type=cli.Choice(TOURNAMENT_ALL_METRICS),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of papers to process per model"),
    ] = 10,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for the tournament to ensure reproducibility"),
    ] = 0,
) -> None:
    """Run a pairwise Elo tournament between multiple models."""
    tournament_metrics = metrics or TOURNAMENT_ALL_METRICS

    dotenv.load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse input files, types, and model names
    parsed_inputs: list[tuple[Path, str]] = []
    model_names: list[str] = []

    for input_str in inputs:
        parts = input_str.split(":", maxsplit=2)
        if len(parts) == 3:
            file_path, file_type, model_name = parts
        elif len(parts) == 2:
            file_path, file_type = parts
            model_name = Path(file_path).parent.name
        else:
            file_path = parts[0]
            file_type = "graph"
            model_name = Path(file_path).parent.name

        parsed_inputs.append((Path(file_path), file_type))
        model_names.append(model_name)

    asyncio.run(
        _run_tournament(
            parsed_inputs,
            model_names,
            output_dir,
            model,
            tournament_prompt,
            tournament_metrics,
            limit,
            seed,
        )
    )


async def _run_tournament(
    inputs: list[tuple[Path, str]],
    model_names: list[str],
    output_dir: Path,
    model: str,
    tournament_prompt_key: str,
    metrics: list[str],
    limit: int,
    seed: int,
) -> None:
    """Run the tournament on the given inputs.

    Args:
        inputs: List of (file_path, file_type) tuples.
        model_names: Names of the models.
        output_dir: Directory to save results.
        model: GPT model to use.
        tournament_prompt_key: Key for the comparison prompt.
        metrics: Metrics to evaluate.
        limit: Maximum number of papers to use.
        seed: Random seed.
    """
    random.seed(seed)

    client = OpenAIClient(
        api_key=ensure_envvar("OPENAI_API_KEY"), model=model, seed=seed
    )

    # Load papers from each input file
    paper_collections: list[Sequence[InputType]] = []

    for file_path, file_type in inputs:
        match file_type.lower():
            case "graph":
                papers = sample(
                    PromptResult.unwrap(
                        load_data(file_path, PromptResult[GraphResult])
                    ),
                    limit,
                )
            case "paper":
                papers = sample(
                    PromptResult.unwrap(
                        load_data(file_path, PromptResult[PaperResult])
                    ),
                    limit,
                )
            case "raw":
                papers = sample(load_data(file_path, pr.Paper), limit)
            case "summ":
                papers = sample(
                    PromptResult.unwrap(
                        load_data(file_path, PromptResult[PaperWithRelatedSummary])
                    ),
                    limit,
                )
            case _:
                raise ValueError(
                    f"Invalid file_type: {file_type}. Must be 'graph', 'paper', 'summ',"
                    " or 'raw'."
                )

        paper_collections.append(papers)

    # Find common papers across all models
    common_papers = find_common_papers(paper_collections)
    logger.info(
        f"Found {len(common_papers)} papers common to all {len(model_names)} models"
    )

    if not common_papers:
        logger.error(
            "No common papers found across all models. Tournament cannot proceed."
        )
        return

    with Timer() as timer:
        results = await run_tournament(
            client,
            common_papers,
            model_names,
            output_dir,
            tournament_prompt_key,
            metrics,
            seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results['total_cost']:.10f}")

    # Display results
    logger.info("\n%s", _display_tournament_results(results))


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
