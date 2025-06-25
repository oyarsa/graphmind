"""Command-line interface for tournament evaluation.

This module provides the CLI for running tournament comparisons of rationales.
"""

from __future__ import annotations

import asyncio
import logging
import random
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated

import dotenv
import typer

from paper.gpt.evaluate_tournament.bradley_terry import calculate_bradley_terry_rankings
from paper.gpt.evaluate_tournament.comparisons import (
    generate_new_comparisons,
    load_reused_comparisons,
)
from paper.gpt.evaluate_tournament.elo import (
    MELO_DEFAULT_TRIALS,
    calculate_elo_rankings,
    calculate_melo_rankings,
)
from paper.gpt.evaluate_tournament.tournament import (
    TOURNAMENT_METRICS,
    InputFileType,
    calculate_token_statistics,
    count_head_to_head,
    display_head_to_head,
    display_tournament_ranks,
    display_tournament_ratings,
    tournament_summary,
)
from paper.gpt.model import PromptResult
from paper.gpt.prompts import load_prompts, print_prompts
from paper.gpt.run_gpt import LLMClient
from paper.util import (
    Timer,
    cli,
    get_params,
    render_params,
    setup_logging,
)
from paper.util.serde import save_data

logger = logging.getLogger(__name__)

# Load prompts for pairwise comparison
PAIRWISE_COMPARISON_PROMPTS = load_prompts("pairwise_comparison")


class RankingAlgorithm(StrEnum):
    """Available ranking algorithms."""

    ELO = "elo"
    MELO = "melo"
    BRADLEY_TERRY = "bradley-terry"


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(no_args_is_help=True)
def run(
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
            "-M",
            help="Metrics to evaluate in tournament",
            click_type=cli.Choice(TOURNAMENT_METRICS),
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
    algorithm: Annotated[
        RankingAlgorithm,
        typer.Option(
            "--algo",
            help="Ranking algorithm to use: 'elo' (single tournament) or 'melo' (10"
            " tournaments with different order)",
        ),
    ] = RankingAlgorithm.ELO,
    reuse_comparisons: Annotated[
        Path | None,
        typer.Option(
            "--reuse",
            help="Path to raw_comparisons.json file from a previous run to reuse. If"
            " provided, ignores input data and parameters.",
        ),
    ] = None,
    melo_trials: Annotated[
        int, typer.Option(help="If the algorithm is 'melo', how many trials to run.")
    ] = MELO_DEFAULT_TRIALS,
    show_head_to_head: Annotated[
        bool,
        typer.Option(
            "--head-to-head", help="Show head to head scores for all metrics."
        ),
    ] = False,
    markdown_table: Annotated[
        bool,
        typer.Option("--markdown", help="Show tournament results table as Markdown."),
    ] = False,
    save_comparisons: Annotated[
        bool,
        typer.Option("--save/--no-save", help="Save comparison results."),
    ] = True,
) -> None:
    """Run a pairwise tournament between multiple models.

    The inputs are given in the `path:type:name` format. The valid types are:
        - raw: `peerread.Paper` from the original dataset
        - graph: `PromptResult[GraphResult]` from `gpt.evaluate_paper_graph`
        - paper: `PromptResult[PaperResult]` from `gpt.evaluate_paper_scimon`
        - summ: `PromptResult[PaperWithRelatedSummary]` from `gpt.summarise_related_peter.py`

    Examples:
        - full-graph/result.json:graph:GraphMind
        - sans/result.json:graph:Basic
        - scimon/scimon-simple/result.json:paper:Scimon
        - orc_merged.json:raw:Human-ORC
        - split/test.json:summ:Human-Test

    The tournament can use different ranking algorithms:
    - elo: Standard Elo rating system with a single ordering
    - melo: Multiple Elo tournaments with different random orderings. Set the number of
      trials with --melo-trials.
    - bradley_terry: Bradley-Terry model using maximum likelihood estimation to compute
      player strengths based on all match outcomes. Unlike Elo, this is not order-dependent
      and uses a global optimization approach.

    If you provide --reuse with a path to a raw_comparisons.json file, the system will
    skip the LLM comparison phase and just calculate rankings using the existing
    comparison data.

    If using --reuse, use --save/--no-save to determine whether the reused file is
    copied to the output alongside the tournament results. The default is to save. New
    comparison results are always saved.
    """
    params = get_params()
    logger.info(render_params(params))

    tournament_metrics = metrics or list(TOURNAMENT_METRICS)

    dotenv.load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse input files, types, and model names
    parsed_inputs: list[tuple[Path, InputFileType]] = []
    model_names: list[str] = []

    for input_str in inputs:
        parts = input_str.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError("Invalid input string format. Must be path:type:name.")

        file_path = Path(parts[0])
        file_type = InputFileType.from_dirty(parts[1])
        model_name = parts[2]

        parsed_inputs.append((file_path, file_type))
        model_names.append(model_name)

    asyncio.run(
        run_tournaments(
            parsed_inputs,
            model_names,
            output_dir,
            model,
            tournament_prompt,
            tournament_metrics,
            limit,
            seed,
            algorithm,
            reuse_comparisons,
            melo_trials,
            show_head_to_head,
            markdown_table,
            save_comparisons,
        )
    )


async def run_tournaments(
    inputs: list[tuple[Path, InputFileType]],
    model_names: list[str],
    output_dir: Path,
    model: str,
    tournament_prompt_key: str,
    metrics: list[str],
    limit: int,
    seed: int,
    algorithm: RankingAlgorithm,
    reuse_comparisons_path: Path | None,
    melo_trials: int,
    show_head_to_head: bool,
    markdown_table: bool,
    save_comparisons: bool,
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
        algorithm: Ranking algorithm to use (elo or melo).
        reuse_comparisons_path: Optional path to comparisons JSON file to reuse. If
            provided, other information (e.g. input data, model names, GPT model) is
            ignored.
        melo_trials: How many MElo trials to run.
        show_head_to_head: Show head to head scores for all metrics.
        markdown_table: Display results table as Markdown.
        save_comparisons: If comparisons should be saved with the tournament results.
            Only considered if we're reusing a previous file. New comparisons are always
            saved.
    """
    rng = random.Random(seed)
    client = LLMClient.new_env(model=model, seed=seed)

    # Step 1: Either load existing comparisons or generate new ones
    if reuse_comparisons_path is not None:
        raw_comparisons = await load_reused_comparisons(reuse_comparisons_path)
    else:
        prompt = PAIRWISE_COMPARISON_PROMPTS[tournament_prompt_key]
        raw_comparisons = await generate_new_comparisons(
            client,
            inputs,
            model_names,
            metrics,
            TOURNAMENT_METRICS,
            limit,
            model,
            tournament_prompt_key,
            seed,
            algorithm,
            prompt,
            rng,
        )

    logger.info("Loaded %d comparisons", len(raw_comparisons.result.comparisons))

    # Step 2: Display head-to-head results, then calculate rankings
    comparisons = PromptResult.unwrap(raw_comparisons.result.comparisons)

    if show_head_to_head:
        head_to_head = count_head_to_head(comparisons, model_names, metrics)
        logger.info(
            "\n%s",
            display_head_to_head(
                model_names, metrics, head_to_head, markdown=markdown_table
            ),
        )

    # Calculate rankings using the selected algorithm and report results
    with Timer() as ranking_timer:
        match algorithm:
            case RankingAlgorithm.ELO:
                ranker = calculate_elo_rankings
            case RankingAlgorithm.MELO:
                ranker = partial(
                    calculate_melo_rankings, seed=seed, num_trials=melo_trials
                )
            case RankingAlgorithm.BRADLEY_TERRY:
                ranker = calculate_bradley_terry_rankings

        tournament_result = ranker(comparisons, model_names, metrics)

        token_stats = calculate_token_statistics(comparisons)
        summary = tournament_summary(
            tournament_result, model_names, metrics, token_stats
        )

    logger.info(f"Rankings calculation time: {ranking_timer.human}")
    logger.info(f"Total comparison cost: ${raw_comparisons.cost:.10f}")

    logger.info("\n%s", display_tournament_ranks(summary, markdown=markdown_table))
    logger.info("\n%s", display_tournament_ratings(summary, markdown=markdown_table))

    if isinstance(raw_comparisons, PromptResult) or save_comparisons:
        save_data(output_dir / "raw_comparisons.json.zst", raw_comparisons.result)
    save_data(output_dir / f"tournament_results_{algorithm}.json.zst", summary)


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts("RATIONALE TOURNAMENT", PAIRWISE_COMPARISON_PROMPTS, detail=detail)


if __name__ == "__main__":
    app()
