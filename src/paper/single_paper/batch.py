"""CLI for batch novelty-report generation from a JSON list of arXiv titles.

Thin wrapper around `paper.single_paper.api.evaluate_papers`: reads titles
from a file, writes one report per paper plus a `failures.json` for any that
errored — all the actual work happens in `api.py`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from paper.single_paper.api import (
    EvaluationError,
    PaperEvaluationConfig,
    evaluate_papers,
)
from paper.util import arun_safe, dotenv, setup_logging
from paper.util.serde import save_data

logger = logging.getLogger(__name__)

_TITLES_ADAPTER = TypeAdapter(list[str])


def _slugify(title: str, max_len: int = 60) -> str:
    """Turn a title into a filesystem-safe slug for output filenames."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
    return slug[:max_len] or "untitled"


async def run_batch(
    titles_file: Path,
    output_dir: Path,
    cache_file: Path,
    concurrency: int,
    config: PaperEvaluationConfig,
) -> None:
    """Generate a novelty report for each title in `titles_file`."""
    titles = _TITLES_ADAPTER.validate_json(titles_file.read_bytes())
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = await evaluate_papers(
        titles, config, concurrency=concurrency, cache_path=cache_file
    )

    failures: dict[str, str] = {}
    for i, evaluation in enumerate(evaluations):
        if isinstance(evaluation.result, EvaluationError):
            logger.warning(
                "Failed to process '%s': %s",
                evaluation.title,
                evaluation.result.message,
            )
            failures[evaluation.title] = evaluation.result.message
        else:
            save_data(
                output_dir / f"{i:03d}_{_slugify(evaluation.title)}.json",
                evaluation.result,
            )

    if failures:
        save_data(output_dir / "failures.json", failures)

    logger.info(
        "Processed %d papers: %d succeeded, %d failed",
        len(titles),
        len(titles) - len(failures),
        len(failures),
    )


def batch_main(
    titles_file: Annotated[
        Path, typer.Argument(help="JSON file with a list of arXiv paper titles")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for novelty reports")
    ],
    cache_file: Annotated[
        Path | None,
        typer.Option(
            "--cache", help="S2 metadata cache file (default: <output>/s2_cache.json)"
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option(help="Maximum papers processed concurrently")
    ] = 4,
    top_k_refs: Annotated[
        int, typer.Option(help="Number of top references to process by similarity")
    ] = 20,
    num_recommendations: Annotated[
        int, typer.Option(help="Number of recommended papers to fetch from S2 API")
    ] = 30,
    num_related: Annotated[
        int, typer.Option(help="Number of related papers per type (positive/negative)")
    ] = 2,
    llm_model: Annotated[
        str, typer.Option(help="GPT/Gemini model to use for API calls")
    ] = "gpt-4o-mini",
    encoder_model: Annotated[
        str, typer.Option(help="OpenAI embedding model for similarity")
    ] = "text-embedding-3-small",
    seed: Annotated[int, typer.Option(help="Random seed for GPT API calls")] = 0,
    eval_prompt: Annotated[
        str, typer.Option(help="User prompt for paper evaluation")
    ] = "full-graph-structured",
    graph_prompt: Annotated[
        str, typer.Option(help="User prompt for graph extraction")
    ] = "full",
    demonstrations: Annotated[
        str, typer.Option(help="Demonstrations file for few-shot prompting")
    ] = "orc_4",
    demo_prompt: Annotated[
        str, typer.Option(help="Demonstration prompt key")
    ] = "abstract",
) -> None:
    """Generate novelty reports for a JSON list of arXiv paper titles.

    Reuses one Semantic Scholar cache across the whole batch, so references and
    recommendations shared between papers are only fetched once. Titles that
    fail (not found on S2/arXiv, LaTeX parse errors, etc.) are recorded in
    `failures.json` under the output directory instead of aborting the batch.
    """
    setup_logging()
    dotenv.load_dotenv()
    config = PaperEvaluationConfig(
        top_k_refs=top_k_refs,
        num_recommendations=num_recommendations,
        num_related=num_related,
        llm_model=llm_model,
        encoder_model=encoder_model,
        seed=seed,
        eval_prompt_key=eval_prompt,
        graph_prompt_key=graph_prompt,
        demonstrations_key=demonstrations,
        demo_prompt_key=demo_prompt,
    )
    arun_safe(
        run_batch,
        titles_file,
        output_dir,
        cache_file or output_dir / "s2_cache.json",
        concurrency,
        config,
    )
