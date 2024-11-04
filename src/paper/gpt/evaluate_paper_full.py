"""Evaluate a paper's approval based on its full-body text."""

# Best configuration:
#     Command:
#     $ uv run gpt eval_full run output/asap_balanced_50.json tmp/eval-full -n 0 \
#         --clean-run --user-prompt simple-abs --demos output/demonstrations_10.json \
#         --demo-prompt simple -m 4o
#
#     2024-11-04 20:03:37 | INFO | paper.gpt.evaluate_paper_full:151 | CONFIG:
#     - model: 4o
#     - api_key: None
#     - data_path: /Users/italo/dev/paper-hypergraph/output/asap_balanced_50.json (dc592a4f)
#     - limit_papers: 0
#     - user_prompt_key: simple-abs
#     - output_dir: /Users/italo/dev/paper-hypergraph/tmp/eval-full (directory)
#     - continue_papers_file: None
#     - clean_run: True
#     - seed: 0
#     - demonstrations_file: /Users/italo/dev/paper-hypergraph/output/demonstrations_10.json (55baa321)
#     - demo_prompt_key: simple
#
# Output:
#     - P   : 0.6286
#     - R   : 0.8800
#     - F1  : 0.7333
#     - Acc : 0.6800
#
#     Gold (P/N): 25/25 (50.00%)
#     Pred (P/N): 35/15 (70.00%)

import argparse
import asyncio
import logging
import os
import random
from collections.abc import Sequence
from pathlib import Path

import dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS,
    Demonstration,
    PaperResult,
    calculate_paper_metrics,
    display_metrics,
    format_demonstrations,
)
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
from paper.util import Timer, display_params, setup_logging

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
                args.demos,
                args.demo_prompt,
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
    demonstrations_file: Path | None,
    demo_prompt_key: str,
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
            `_GRAPH_USER_PROMPTS` for available options or `list_prompts` for more.
        classify_user_prompt_key: Key to the user prompt to use for graph extraction. See
            `_CLASSIFY_USER_PROMPTS` for available options or `list_prompts` for more.
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
        demonstrations_file: Path to demonstrations file for use with few-shot prompting.
        demo_prompt_key: Key to the demonstration prompt to use during evaluation to
            build the few-shot prompt. See `EVALUTE_DEMONSTRATION_PROMPTS` for the
            avaialble options or `list_prompts` for more.

    Returns:
        None. The output is saved to `output_dir`.
    """
    logger.info(display_params())

    random.seed(seed)

    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = AsyncOpenAI()

    data = TypeAdapter(list[Paper]).validate_json(data_path.read_bytes())
    random.shuffle(data)

    papers = data[:limit_papers]
    demonstrations = (
        TypeAdapter(list[Demonstration]).validate_json(demonstrations_file.read_bytes())
        if demonstrations_file is not None
        else []
    )
    user_prompt = _FULL_CLASSIFY_USER_PROMPTS[user_prompt_key]
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]

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
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    logger.info(
        "Skipping %d items from the `continue` file.", len(papers_remaining.done)
    )

    with Timer() as timer:
        results = await _classify_papers(
            client,
            model,
            user_prompt,
            papers_remaining.remaining,
            output_intermediate_file,
            demonstrations,
            demonstration_prompt,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = [result.item for result in results_all]

    metrics = calculate_paper_metrics(results_items)
    logger.info(display_metrics(metrics, results_items) + "\n")

    assert len(results_all) == len(papers)
    (output_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperResult]]).dump_json(results_all, indent=2)
    )
    (output_dir / "result_items.json").write_bytes(
        TypeAdapter(list[PaperResult]).dump_json(results_items, indent=2)
    )
    (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))


async def _classify_papers(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[Paper],
    output_intermediate_file: Path,
    demonstrations: Sequence[Demonstration],
    demonstration_prompt: PromptTemplate,
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

    tasks = [
        _classify_paper(
            client, model, paper, user_prompt, demonstrations, demonstration_prompt
        )
        for paper in papers
    ]

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


_FULL_CLASSIFY_SYSTEM_PROMPT = (
    "Give an approval or rejection to a paper submitted to a high-quality scientific"
    " conference."
)


async def _classify_paper(
    client: AsyncOpenAI,
    model: str,
    paper: Paper,
    user_prompt: PromptTemplate,
    demonstrations: Sequence[Demonstration],
    demonstration_prompt: PromptTemplate,
) -> GPTResult[PromptResult[PaperResult]]:
    user_prompt_text = user_prompt.template.format(
        title=paper.title,
        abstract=paper.abstract,
        main_text=paper.main_text(),
        demonstrations=format_demonstrations(demonstrations, demonstration_prompt),
    )
    result = await run_gpt(
        _CLASSIFY_TYPES[user_prompt.type_name],
        client,
        _FULL_CLASSIFY_SYSTEM_PROMPT,
        user_prompt_text,
        model,
    )
    classified = result.result

    return GPTResult(
        result=PromptResult(
            item=PaperResult(
                title=paper.title,
                abstract=paper.abstract,
                reviews=paper.reviews,
                sections=paper.sections,
                approval=paper.approval,
                y_true=paper.is_approved(),
                y_pred=classified.approved if classified else False,
                rationale=classified.rationale if classified else "<error>",
            ),
            prompt=Prompt(system=_FULL_CLASSIFY_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


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
    run_parser.add_argument(
        "--demos",
        type=Path,
        default=None,
        help="File containing demonstrations to use in few-shot prompt",
    )
    run_parser.add_argument(
        "--demo-prompt",
        type=str,
        choices=EVALUATE_DEMONSTRATION_PROMPTS.keys(),
        default="simple",
        help="The user prompt to use for building the few-shot demonstrations. Defaults"
        " to %(default)s.",
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


if __name__ == "__main__":
    main()
