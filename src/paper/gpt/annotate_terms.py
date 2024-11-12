"""Extract key terms for problems and methods from S2 Papers.

Input is a JSON array of `paper.external_data.semantic_scholar.model.Paper`. Output
contains the input paper plus the prompts used and the extracted terms.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

import typer
from openai import AsyncClient, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from paper import progress
from paper.external_data.semantic_scholar.model import Paper
from paper.gpt.model import Prompt, PromptResult
from paper.gpt.prompts import load_prompts, print_prompts
from paper.gpt.run_gpt import MODELS_ALLOWED, GPTResult, run_gpt
from paper.util import (
    HelpOnErrorArgumentParser,
    load_data,
    mustenv,
    save_data,
    setup_logging,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


_SYSTEM_PROMPT = """\
You are a helpful assistant that can read scientific papers and identify the key terms \
used to describe the problems tackled by the paper and the terms used for the methods.
"""
_ANN_USER_PROMPTS = load_prompts("annotate_terms")


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    setup_cli_parser(parser)

    args = parser.parse_args()
    setup_logging()

    if args.subcommand == "prompts":
        list_prompts(detail=args.detail)
    elif args.subcommand == "run":
        asyncio.run(
            annotate_papers_terms(
                args.input_file,
                args.output_file,
                args.limit,
                args.model,
                args.seed,
                args.user_prompt,
            )
        )


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", required=True
    )

    # 'run' subcommand parser
    run_parser = subparsers.add_parser(
        "run",
        help="Run the term annotation process",
        description="Run the term annotation process with the provided arguments.",
    )

    # Add original arguments to the 'run' subcommand
    run_parser.add_argument(
        "input_file",
        type=Path,
        help="The path to the JSON file containing the papers data (S2 `Paper` format).",
    )
    run_parser.add_argument(
        "output_file",
        type=Path,
        help="The path to the output JSON file with the annotated terms.",
    )
    run_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        choices=MODELS_ALLOWED,
        help="The model to use for the annotation.",
    )
    run_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=1,
        help="The number of papers to process. Defaults to 1. Set to 0 for all papers.",
    )
    run_parser.add_argument(
        "--seed", default=0, type=int, help="Seed to set in OpenAI call"
    )
    run_parser.add_argument(
        "--user-prompt",
        type=str,
        choices=sorted(_ANN_USER_PROMPTS),
        default="sentence",
        help="The user prompt to use for context classification.",
    )

    # 'prompts' subcommand parser
    prompts_parser = subparsers.add_parser(
        "prompts",
        help="List available prompts",
        description="List available prompts. Use --detail for more information.",
    )
    prompts_parser.add_argument(
        "--detail",
        action="store_true",
        help="Provide detailed descriptions of the prompts.",
    )


async def annotate_papers_terms(
    input_file: Path,
    output_file: Path,
    limit_papers: int | None,
    model: str,
    seed: int,
    user_prompt_key: str,
) -> None:
    """Extract problem and method terms from each paper.

    Args:
        input_file: JSON with input data.
            Array of `paper.external_data.semantic_scholar.model.Paper`.
        output_file: File to save the output JSON data. See `PaperTermAnnotated`.
        limit_papers: How many papers to process. 0 or None means all papers.
        model: Name of the OpenAI API model to use.
            See `papers.gpt.run_gpt.MODELS_ALLOWED` for the allowed ones.
        seed: Random generator seed to pass to the model to try to get some
            reproducibility.
        user_prompt_key: Key of the prompt to use in the annotation prompt mapping.
    """
    env = mustenv("OPENAI_API_CLIENT")
    client = AsyncOpenAI(api_key=env["OPENAI_API_CLIENT"])

    if limit_papers == 0:
        limit_papers = None

    data = load_data(input_file, Paper)[:limit_papers]
    output = await _annotate_papers_terms(client, model, seed, data, user_prompt_key)

    logger.info(f"Total cost: ${output.cost:.10f}")
    save_data(output_file, output.result)


class PaperAnnotatedTerms(BaseModel):
    """S2 Paper with its annotated key terms. Includes GPT prompts used."""

    model_config = ConfigDict(frozen=True)

    terms: PromptResult[GPTTerms]
    paper: Paper


class GPTTerms(BaseModel):
    """Terms used to describe the paper problems and the applied methods."""

    model_config = ConfigDict(frozen=True)

    problem: Sequence[str] = Field(
        description="Terms used to describe the problem tackled by the paper."
    )
    methods: Sequence[str] = Field(
        description="Terms used to describe the methods used to solve the paper"
        " problem."
    )


async def _annotate_papers_terms(
    client: AsyncClient,
    model: str,
    seed: int,
    papers: Sequence[Paper],
    user_prompt_key: str,
) -> GPTResult[list[PaperAnnotatedTerms]]:
    """Annotate papers to add key terms. Runs multiple tasks concurrently."""
    tasks = [
        _annotate_paper_term_single(client, model, seed, paper, user_prompt_key)
        for paper in papers
    ]
    results = await progress.gather(tasks, desc="Extracting paper terms")

    output: list[PaperAnnotatedTerms] = []
    total_cost = 0

    for terms, paper in zip(results, papers):
        output.append(PaperAnnotatedTerms(terms=terms.result, paper=paper))
        total_cost += terms.cost

    return GPTResult(result=output, cost=total_cost)


async def _annotate_paper_term_single(
    client: AsyncClient, model: str, seed: int, paper: Paper, user_prompt_key: str
) -> GPTResult[PromptResult[GPTTerms]]:
    """Annotate a single paper with its key terms."""
    user_prompt_text = _ANN_USER_PROMPTS[user_prompt_key].template.format(
        title=paper.title, abstract=paper.abstract
    )
    result = await run_gpt(
        GPTTerms,
        client,
        _SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )
    terms = result.result if result.result else GPTTerms(problem=[], methods=[])
    return GPTResult(
        result=PromptResult(
            item=terms, prompt=Prompt(user=user_prompt_text, system=_SYSTEM_PROMPT)
        ),
        cost=result.cost,
    )


def list_prompts(detail: bool) -> None:
    print_prompts("TERM ANNOTATION PROMPTS", _ANN_USER_PROMPTS, detail=detail)

    app()
