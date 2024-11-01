"""Evaluate a paper's approval based on its full-body text."""

import argparse
import asyncio
import hashlib
import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path

import dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from paper.evaluation_metrics import Metrics
from paper.gpt.evaluate_paper import PaperResult, calculate_paper_metrics
from paper.gpt.model import Paper, Prompt, PromptResult
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    append_intermediate_result,
    get_id,
    get_remaining_items,
    run_gpt,
)
from paper.progress import as_completed
from paper.util import Timer, safediv, setup_logging

logger = logging.getLogger("paper.gpt.evaluate_paper_full")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    setup_cli_parser(parser)

    args = parser.parse_args()
    setup_logging()

    if args.subcommand == "prompts":
        list_prompts(detail=args.detail)
    elif args.subcommand == "run":
        asyncio.run(
            evaluate_papers(
                args.model,
                args.api_key,
                args.data_path,
                args.limit,
                args.user_prompt,
                args.output_dir,
                args.continue_papers,
                args.clean_run,
                args.seed,
            )
        )


_FULL_CLASSIFY_USER_PROMPTS = load_prompts("evaluate_paper_full")


async def evaluate_papers(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    clean_run: bool,
    seed: int,
) -> None:
    """Evaluate a paper's approval based on its full-body text.

    The papers should come from the ASAP-Review dataset as processed by the
    paper.asap module.

    The classification part is optional. It uses the generated graphs as input as saves
    the results (metrics and predictions) to {output_dir}/classification.

    Args:
        model: GPT model code. Must support Structured Outputs.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        data_path: Path to the JSON file containing the input papers data.
        limit: Number of papers to process. Defaults to 1 example. If None, process all.
        graph_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_GRAPH_USER_PROMPTS` for available options or `_display_prompts` for more.
        classify_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_CLASSIFY_USER_PROMPTS` for available options or `_display_prompts` for more.
        display: If True, show each graph on screen. This suspends the process until
            the plot is closed.
        output_dir: Directory to save the output files: serialised graphs (GraphML),
            plot images (PNG) and classification results (JSON), if classification is
            enabled.
        classify: If True, classify the papers based on the generated graph.
        continue_papers_file: If provided, check for entries in the input data. If they
            are there, we use those results and skip processing them.
        clean_run: If True, ignore `continue_papers` and run everything from scratch.
        seed: Random seed used for shuffling.

    Returns:
        None. The output is saved to disk.
    """
    random.seed(seed)

    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    _log_config(
        model=model,
        data_path=data_path,
        limit_papers=limit_papers,
        user_prompt=user_prompt_key,
        output_dir=output_dir,
        continue_papers_file=continue_papers_file,
        clean_run=clean_run,
    )

    client = AsyncOpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_bytes())
    random.shuffle(data)

    papers = data[:limit_papers]
    user_prompt = _FULL_CLASSIFY_USER_PROMPTS[user_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperResult,
        output_intermediate_file,
        continue_papers_file,
        papers,
        clean_run=clean_run,
        continue_key=get_id,
        original_key=get_id,
    )
    if not papers_remaining.remaining:
        logging.warning(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    logging.warning(
        "Skipping %d items from the `continue` file.", len(papers_remaining.done)
    )

    with Timer() as timer:
        results = await _classify_papers(
            client,
            model,
            user_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = [result.item for result in results_all]

    metrics = calculate_paper_metrics(results_items)
    logger.info(_display_metrics(metrics, results_items))

    assert len(results_all) == len(papers)
    (output_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperResult]]).dump_json(results_all, indent=2)
    )
    (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[Paper],
    output_intermediate_file: Path,
) -> GPTResult[list[PromptResult[PaperResult]]]:
    """Classify Papers into approved/not approved using the paper main text.

    Args:
        client: OpenAI client to use GPT
        model: GPT model code to use (must support Structured Outputs)
        user_prompt: User prompt template to use for classification to be filled
        papers: Papers from the ASAP-Review dataset to classify
        output_intermediate_file: File to write new results after each task is completed

    Returns:
        List of classified papers wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperResult]] = []
    total_cost = 0

    tasks = [_classify_paper(client, model, paper, user_prompt) for paper in papers]

    for task in as_completed(tasks, desc="Classifying papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        append_intermediate_result(PaperResult, output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


class GPTFull(BaseModel):
    """Decision on if the paper should be published and the reason for the decision."""

    model_config = ConfigDict(frozen=True)

    rationale: str = Field(description="How you reached your approval decision.")
    approved: bool = Field(description="If the paper was approved for publication.")


_CLASSIFY_TYPES = {
    "full": GPTFull,
}


FULL_CLASSIFY_SYSTEM_PROMPT = (
    "Approve or reject the scientific paper based on the following paper text."
)


async def _classify_paper(
    client: AsyncOpenAI, model: str, paper: Paper, user_prompt: PromptTemplate
) -> GPTResult[PromptResult[PaperResult]]:
    user_prompt_text = user_prompt.template.format(
        title=paper.title, abstract=paper.abstract, main_text=paper.main_text()
    )
    result = await run_gpt(
        _CLASSIFY_TYPES[user_prompt.type_name],
        client,
        FULL_CLASSIFY_SYSTEM_PROMPT,
        user_prompt_text,
        model,
    )
    classified = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                ratings=paper.ratings,
                sections=paper.sections,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
                approval=paper.approval,
            ),
            prompt=Prompt(system=FULL_CLASSIFY_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


def _display_metrics(metrics: Metrics, results: Sequence[PaperResult]) -> str:
    y_true = [r.y_true for r in results]
    y_pred = [r.y_pred for r in results]

    output = [
        "Metrics:",
        str(metrics),
        "",
        f"Gold (P/N): {sum(y_true)}/{len(y_true) - sum(y_true)}"
        f" ({safediv(sum(y_true), len(y_true)):.2%})",
        f"Pred (P/N): {sum(y_pred)}/{len(y_pred) - sum(y_pred)}"
        f" ({safediv(sum(y_pred), len(y_pred)):.2%})",
    ]
    return "\n".join(output)


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        dest="subcommand",
        required=True,
    )

    # 'run' subcommand parser
    run_parser = subparsers.add_parser(
        "run",
        help="Run paper classification",
        description="Run paper classification with the provided arguments.",
    )

    # Add original arguments to the 'run' subcommand
    run_parser.add_argument(
        "data_path",
        type=Path,
        help="The path to the JSON file containing the papers data.",
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
        help="The model to use for the extraction. Defaults to %(default)s.",
    )
    run_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "The OpenAI API key to use for the extraction. Defaults to OPENAI_API_KEY"
            " env var. Can be read from the .env file."
        ),
    )
    run_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=1,
        help="The number of papers to process. Defaults to %(default)s example.",
    )
    run_parser.add_argument(
        "--user-prompt",
        type=str,
        choices=_FULL_CLASSIFY_USER_PROMPTS.keys(),
        default="simple",
        help="The user prompt to use for paper classification. Defaults to"
        " %(default)s.",
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
    run_parser.add_argument(
        "--seed", default=0, type=int, help="Random seed used for shuffling."
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


def list_prompts(detail: bool) -> None:
    print_prompts("FULL PAPER EVALUATION", _FULL_CLASSIFY_USER_PROMPTS, detail=detail)


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit_papers: int | None,
    user_prompt: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    clean_run: bool,
) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    logger.info(
        "CONFIG:\n"
        f"  Model: {model}\n"
        f"  Data path: {data_path.resolve()}\n"
        f"  Data hash (sha256): {data_hash}\n"
        f"  Output dir: {output_dir.resolve()}\n"
        f"  Limit papers: {limit_papers if limit_papers is not None else 'All'}\n"
        f"  User prompt: {user_prompt}\n"
        f"  Continue papers file: {continue_papers_file}\n"
        f"  Clean run: {clean_run}\n"
    )


if __name__ == "__main__":
    main()
