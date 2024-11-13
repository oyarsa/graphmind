"""Extract key terms for problems and methods from S2 Papers.

Input is a JSON array of `paper.external_data.semantic_scholar.model.Paper`. Output
contains the input paper plus the prompts used and the extracted terms.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self, override

import dotenv
from openai import AsyncClient, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from paper import progress
from paper.external_data.semantic_scholar.model import Paper
from paper.gpt.model import Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    append_intermediate_result,
    get_remaining_items,
    run_gpt,
)
from paper.util import (
    HelpOnErrorArgumentParser,
    Record,
    Timer,
    display_params,
    load_data,
    mustenv,
    save_data,
    setup_logging,
)

logger = logging.getLogger(__name__)


_TERM_SYSTEM_PROMPT = """\
You are a helpful assistant that can read scientific papers and identify the key terms \
used to describe the problems tackled by the paper and the terms used for the methods.
"""
_TERM_USER_PROMPTS = load_prompts("annotate_terms")


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
                args.output_dir,
                args.limit,
                args.model,
                args.seed,
                args.user_prompt,
                args.continue_papers,
                args.clean_run,
            )
        )


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", required=True
    )

    # 'run' subcommand parser
    run_parser = subparsers.add_parser(
        "run", help="Run the term annotation process", description=__doc__
    )

    # Add original arguments to the 'run' subcommand
    run_parser.add_argument(
        "input_file",
        type=Path,
        help="The path to the JSON file containing the papers data (S2 `Paper` format).",
    )
    run_parser.add_argument(
        "output_dir",
        type=Path,
        help="The path to the output directory where files will be saved.",
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
        choices=sorted(_TERM_USER_PROMPTS),
        default="simple",
        help="The user prompt to use term annotation.",
    )
    run_parser.add_argument(
        "--continue-papers",
        type=Path,
        default=None,
        help="Path to file with data from a previous run",
    )
    run_parser.add_argument(
        "--clean-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start from scratch, ignoring existing intermediate results",
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
    output_dir: Path,
    limit_papers: int | None,
    model: str,
    seed: int,
    user_prompt_key: str,
    continue_papers_file: Path | None,
    clean_run: bool,
) -> None:
    """Extract problem and method terms from each paper.

    Args:
        input_file: JSON with input data.
            Array of `paper.external_data.semantic_scholar.model.Paper`.
        output_dir: Directory to save output files, including final and intermedaite
            results.
        limit_papers: How many papers to process. 0 or None means all papers.
        model: Name of the OpenAI API model to use.
            See `papers.gpt.run_gpt.MODELS_ALLOWED` for the allowed ones.
        seed: Random generator seed to pass to the model to try to get some
            reproducibility.
        user_prompt_key: Key of the prompt to use in the annotation prompt mapping.
        continue_papers_file: File with the intermediate results from a previous run
            that we want to continue.
        clean_run: If True, we ignore `continue_papers_file` and start from scratch.
    """
    logger.info(display_params())

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    env = mustenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=env["OPENAI_API_KEY"])
    user_prompt = _TERM_USER_PROMPTS[user_prompt_key]
    type_ = _TERM_TYPES[user_prompt.type_name]

    papers = load_data(input_file, Paper)[:limit_papers]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperAnnotatedTerms[type_],
        output_intermediate_file,
        continue_papers_file,
        papers,
        clean_run,
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    if clean_run:
        logger.info("Clean run: ignoring `continue` file and using the whole data.")
    else:
        logger.info(
            "Skipping %d items from the `continue` file.", len(papers_remaining.done)
        )

    papers = load_data(input_file, Paper)[:limit_papers]
    with Timer() as timer:
        output = await _annotate_papers_terms(
            client,
            model,
            papers,
            user_prompt,
            output_intermediate_file,
            type_,
            seed=seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${output.cost:.10f}")

    save_data(output_dir / "results.json", output.result)
    assert len(papers) == len(output.result)


class PaperAnnotatedTerms[T: GPTTermBase](Record):
    """S2 Paper with its annotated key terms. Includes GPT prompts used."""

    terms: T
    paper: Paper

    @property
    def id(self) -> int:
        return self.paper.id


class GPTTermBase(BaseModel, ABC):
    @classmethod
    @abstractmethod
    def empty(cls) -> Self: ...


class GPTSimpleTerms(GPTTermBase):
    """Terms used to describe the paper problems and the applied methods."""

    model_config = ConfigDict(frozen=True)

    problem: Sequence[str] = Field(
        description="Terms used to describe the problem tackled by the paper."
    )
    methods: Sequence[str] = Field(
        description="Terms used to describe the methods used to solve the paper"
        " problem."
    )

    @override
    @classmethod
    def empty(cls) -> Self:
        return cls(problem=[], methods=[])


class GPTTermRelation(BaseModel):
    """Represents a relation between two scientific terms."""

    term1: str
    relation_type: str
    term2: str


class GPTMultiTerms(GPTTermBase):
    """Structured output for scientific term extraction."""

    tasks: Sequence[str]
    methods: Sequence[str]
    metrics: Sequence[str]
    resources: Sequence[str]
    relations: Sequence[GPTTermRelation]

    @override
    @classmethod
    def empty(cls) -> Self:
        return cls(tasks=[], methods=[], metrics=[], resources=[], relations=[])


_TERM_TYPES: Mapping[str, type[GPTTermBase]] = {
    "simple-terms": GPTSimpleTerms,
    "multi-terms": GPTMultiTerms,
}


async def _annotate_papers_terms[T: GPTTermBase](
    client: AsyncClient,
    model: str,
    papers: Sequence[Paper],
    user_prompt: PromptTemplate,
    output_intermediate_path: Path,
    type_: type[T],
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperAnnotatedTerms[T]]]]:
    """Annotate papers to add key terms. Runs multiple tasks concurrently."""
    ann_outputs: list[PromptResult[PaperAnnotatedTerms[T]]] = []
    total_cost = 0

    tasks = [
        _annotate_paper_term_single(client, model, seed, paper, user_prompt, type_)
        for paper in papers
    ]

    for task in progress.as_completed(tasks, desc="Extracting paper terms"):
        result = await task
        total_cost += result.cost

        ann_outputs.append(result.result)
        append_intermediate_result(
            PaperAnnotatedTerms[type_], output_intermediate_path, result.result
        )

    return GPTResult(result=ann_outputs, cost=total_cost)


async def _annotate_paper_term_single[T: GPTTermBase](
    client: AsyncClient,
    model: str,
    seed: int,
    paper: Paper,
    user_prompt: PromptTemplate,
    type_: type[T],
) -> GPTResult[PromptResult[PaperAnnotatedTerms[T]]]:
    """Annotate a single paper with its key terms."""
    user_prompt_text = user_prompt.template.format(
        title=paper.title, abstract=paper.abstract
    )
    result = await run_gpt(
        type_,
        client,
        _TERM_SYSTEM_PROMPT,
        user_prompt_text,
        model,
        seed=seed,
    )
    terms = result.result if result.result else type_.empty()
    return GPTResult(
        result=PromptResult(
            item=PaperAnnotatedTerms(terms=terms, paper=paper),
            prompt=Prompt(user=user_prompt_text, system=_TERM_SYSTEM_PROMPT),
        ),
        cost=result.cost,
    )


def list_prompts(detail: bool) -> None:
    print_prompts("TERM ANNOTATION PROMPTS", _TERM_USER_PROMPTS, detail=detail)
