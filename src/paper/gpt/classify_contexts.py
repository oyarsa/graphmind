"""Classify paper citation contexts by polarity using GPT-4.

Each paper contains many references. These can appear in one or more citation contexts
inside the text. Each citation context can be classified by polarity (positive vs
negative).

NB: We do concurrent requests (mediated by a rate limiter). Unfortunately, that doesn't
work very well with OpenAI's client. This means you'll likely see a lot of
openai.APIConnectionError thrown around. Most requests will go through, so you'll just
have to run the script again until you get everything. See also the `--continue-papers`
option.
"""

import argparse
import asyncio
import contextlib
import hashlib
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from paper import evaluation_metrics
from paper.asap.model import (
    ContextPolarityBinary,
    PaperSection,
    PaperWithFullReference,
)
from paper.asap.model import PaperWithFullReference as PaperInput
from paper.gpt.prompts import PromptTemplate, load_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    Prompt,
    PromptResult,
    run_gpt,
)
from paper.progress import as_completed
from paper.util import Timer, safediv, setup_logging

logger = logging.getLogger("paper.gpt.classify_contexts")


class ContextClassified(BaseModel):
    """Context from a paper reference with its classified polarity."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Full text of the context mention")
    gold: ContextPolarityBinary | None = Field(
        description="Whether the citation context is annotated as positive or negative."
        " Can be absent for unannotated data."
    )
    prediction: ContextPolarityBinary = Field(
        description="Whether the citation context is predicted positive or negative"
    )


class Reference(BaseModel):
    """Paper reference where its contexts are enriched with polarity."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the citation in the paper references")
    s2title: str = Field(description="Title of the citation in the S2 data")
    year: int = Field(description="Year of publication")
    authors: Sequence[str] = Field(description="Author names")
    abstract: str = Field(description="Abstract text")
    contexts: Sequence[ContextClassified] = Field(
        description="Citation contexts from this reference"
    )


class PaperOutput(BaseModel):
    """Paper where its references have contexts with polarity."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[Reference] = Field(description="References made in the paper")


_CONTEXT_SYSTEM_PROMPT = (
    "Classify the context polarity between the main paper and its citation."
)

_CONTEXT_USER_PROMPTS = load_prompts("classify_contexts")


class GPTContext(BaseModel):
    """Context from a paper reference with GPT-classified polarity.

    NB: This is currently identical to the main type (PaperContextClassified). They're
    separate on purpose.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Full text of the context mention")
    polarity: ContextPolarityBinary = Field(
        description="Whether the citation context is positive or negative"
    )


_CONTEXT_TYPES = {
    "context": GPTContext,
}


async def _classify_paper(
    client: AsyncOpenAI,
    limit_references: int | None,
    model: str,
    paper: PaperWithFullReference,
    user_prompt: PromptTemplate,
) -> GPTResult[PromptResult[PaperOutput]]:
    """Classify the contexts for the paper's references by polarity.

    Polarity (ContextPolarity): positive (supports argument) or negative (counterpoint).

    The returned object `PaperOutput` is very similar to the input `PaperInput`, but
    the references are different. Instead of the contexts being only strings, they
    are now `ContextClassified`, containing the original text plus the predicted
    polarity.
    """

    classified_references: list[Reference] = []
    user_prompt_save = None
    total_cost = 0

    references = paper.references[:limit_references]
    for reference in references:
        classified_contexts: list[ContextClassified] = []

        for context in reference.contexts:
            user_prompt_text = user_prompt.template.format(
                main_title=paper.title,
                main_abstract=paper.abstract,
                reference_title=reference.s2title,
                reference_abstract=reference.abstract,
                context=context.sentence,
            )
            if not user_prompt_save:
                user_prompt_save = user_prompt_text

            result = await run_gpt(
                _CONTEXT_TYPES[user_prompt.type_name],
                client,
                _CONTEXT_SYSTEM_PROMPT,
                user_prompt_text,
                model,
            )
            total_cost += result.cost

            if gpt_context := result.result:
                classified_contexts.append(
                    ContextClassified(
                        text=context.sentence,
                        gold=ContextPolarityBinary.from_trinary(context.polarity)
                        if context.polarity is not None
                        else None,
                        prediction=gpt_context.polarity,
                    )
                )

        classified_references.append(
            Reference(
                title=reference.title,
                year=reference.year,
                authors=reference.authors,
                abstract=reference.abstract,
                s2title=reference.s2title,
                contexts=classified_contexts,
            )
        )

    # Some references might have fewer contexts after classification, but all
    # references should be in the output.
    if len(classified_references) != len(references):
        logger.warning(
            "Paper %r has %d references but only %d were classified.",
            paper.title,
            len(references),
            len(classified_references),
        )

    result = PromptResult(
        prompt=Prompt(system=_CONTEXT_SYSTEM_PROMPT, user=user_prompt_save or ""),
        item=PaperOutput(
            title=paper.title,
            abstract=paper.abstract,
            ratings=paper.ratings,
            sections=paper.sections,
            approval=paper.approval,
            references=classified_references,
        ),
    )
    return GPTResult(result, total_cost)


async def _classify_contexts(
    client: AsyncOpenAI,
    model: str,
    user_prompt: PromptTemplate,
    papers: Sequence[PaperInput],
    limit_references: int | None,
    output_intermediate_path: Path,
) -> GPTResult[list[PromptResult[PaperOutput]]]:
    """Classify the contexts for each papers' references by polarity.

    Polarity (ContextPolarity): positive (supports argument) or negative (counterpoint).

    The returned object `PaperOutput` is very similar to the input `PaperInput`, but the
    references are different. Instead of the contexts being only strings, they are now
    `ContextClassified`, containing the original text plus the predicted polarity.

    Requests to the OpenAI API are made concurrently, respecting the rate limits. As
    each request is completed, the result is saved to an intermediate file. Once all
    are done, a full result is returned.
    """
    paper_outputs: list[PromptResult[PaperOutput]] = []
    total_cost = 0

    tasks = [
        _classify_paper(client, limit_references, model, paper, user_prompt)
        for paper in papers
    ]
    for task in as_completed(tasks, desc="Classifying paper reference contexts"):
        result = await task
        total_cost += result.cost

        paper_outputs.append(result.result)
        _append_intermediate_result(output_intermediate_path, result.result)

    return GPTResult(paper_outputs, total_cost)


def _append_intermediate_result(path: Path, result: PromptResult[PaperOutput]) -> None:
    """Save result to intermediate file.

    If the intermediate file doesn't exist, create a new one containing the result.
    """
    result_adapter = TypeAdapter(list[PromptResult[PaperOutput]])

    previous = []
    try:
        previous = result_adapter.validate_json(path.read_bytes())
    except FileNotFoundError:
        # It's fine if the file didn't exist previously. We'll create a new one now.
        pass
    except Exception:
        logger.exception("Error reading intermediate result file: %s", path)

    previous.append(result)
    try:
        path.write_bytes(result_adapter.dump_json(previous, indent=2))
    except Exception:
        logger.exception("Error writing intermediate results to: %s", path)


def _paper_id(paper: PaperOutput | PaperWithFullReference) -> int:
    return hash(paper.title + paper.abstract)


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit_papers: int | None,
    limit_references: int | None,
    user_prompt: str,
    output_dir: Path,
) -> None:
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    logger.info(
        "CONFIG:\n"
        f"  Model: {model}\n"
        f"  Data path: {data_path.resolve()}\n"
        f"  Data hash (sha256): {data_hash}\n"
        f"  Output dir: {output_dir.resolve()}\n"
        f"  Limit papers: {limit_papers if limit_papers is not None else 'All'}\n"
        f"  Limit references: {
            limit_references if limit_references is not None else 'All'
        }\n"
        f"  User prompt: {user_prompt}\n"
    )


async def classify_contexts(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    limit_references: int | None,
    continue_papers_file: Path | None,
) -> None:
    """Classify reference citation contexts by polarity."""

    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    if limit_references == 0:
        limit_references = None

    _log_config(
        model=model,
        data_path=data_path,
        limit_papers=limit_papers,
        limit_references=limit_references,
        user_prompt=user_prompt_key,
        output_dir=output_dir,
    )

    client = AsyncOpenAI()

    data = TypeAdapter(list[PaperInput]).validate_json(data_path.read_bytes())

    papers = data[:limit_papers]
    user_prompt = _CONTEXT_USER_PROMPTS[user_prompt_key]

    result_adapter = TypeAdapter(list[PromptResult[PaperOutput]])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_path = output_dir / "results.tmp.json"

    if continue_papers_file is None and output_intermediate_path.is_file():
        continue_papers_file = output_intermediate_path

    continue_papers = []
    if continue_papers_file:
        logger.info("Continuing papers from: %s", continue_papers_file)
        with contextlib.suppress(Exception):
            continue_papers = result_adapter.validate_json(
                continue_papers_file.read_bytes()
            )

    continue_paper_ids = {_paper_id(paper.item) for paper in continue_papers}
    papers_num = len(papers)
    papers = [paper for paper in papers if _paper_id(paper) not in continue_paper_ids]
    if not papers:
        logger.warning(
            "No remaining papers to classify. They're all on the intermediate results."
        )
        return
    else:
        logger.info("Skipping %d papers.", papers_num - len(papers))

    with Timer() as timer:
        results = await _classify_contexts(
            client,
            model,
            user_prompt,
            papers,
            limit_references,
            output_intermediate_path,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    contexts = [result.item for result in results.result]
    stats, metrics = show_classified_stats(contexts)
    logger.info("Classification metrics:\n%s\n", stats)

    (output_dir / "result.json").write_bytes(
        result_adapter.dump_json(results.result, indent=2)
    )
    (output_dir / "output.txt").write_text(stats)
    if metrics is not None:
        (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))


async def on_result_completion(
    result: PromptResult[PaperOutput], lock: asyncio.Lock, path: Path
) -> None:
    adapter = TypeAdapter(list[PromptResult[PaperOutput]])
    async with lock:
        previous = []
        try:
            previous = adapter.validate_json(path.read_bytes())
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Error reading intermediate result file")

        previous.append(result)
        path.write_bytes(adapter.dump_json(previous, indent=2))


# TODO:This one is a bit messy. Refactor it.
def show_classified_stats(
    data: Iterable[PaperOutput],
) -> tuple[str, evaluation_metrics.Metrics | None]:
    """Evaluate the annotation results and print statistics.

    If the data includes gold annotation, calculate evaluation metrics. If not,
    render frequency statistics.

    Returns:
        If the data includes gold annotation, render evaluation metrics and frequency
        pred/gold frequency information. Also return the calculated metrics.
        Otherwise, render basic frequency information and don't return metrics.
    """
    all_contexts: list[ContextClassified] = []
    y_true: list[bool] = []
    y_pred: list[bool] = []

    for paper in data:
        for reference in paper.references:
            for context in reference.contexts:
                all_contexts.append(context)
                if context.gold is not None:
                    y_true.append(context.gold is ContextPolarityBinary.POSITIVE)
                    y_pred.append(context.prediction is ContextPolarityBinary.POSITIVE)

    output = [
        f"Total contexts: {len(all_contexts)}",
        "",
    ]

    if y_true:
        metrics = evaluation_metrics.calculate_metrics(y_true, y_pred)
        output += [
            str(metrics),
            "",
            f"Gold (P/N): {sum(y_true)}/{len(y_true) - sum(y_true)}"
            f" ({safediv(sum(y_true),len(y_true)):.2%})",
            f"Pred (P/N): {sum(y_pred)}/{len(y_pred) - sum(y_pred)}"
            f" ({safediv(sum(y_pred),len(y_pred)):.2%})",
        ]
        return "\n".join(output), metrics

    # No entries with gold annotation
    positive = sum(
        context.prediction is ContextPolarityBinary.POSITIVE for context in all_contexts
    )
    negative = len(all_contexts) - positive
    output += [
        "No gold values to calculate metrics.",
        f"Positive: {positive} ({safediv(positive , len(all_contexts)):.2%})",
        f"Negative {negative} ({safediv(negative , len(all_contexts)):.2%})",
    ]
    return "\n".join(output), None


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
        help="Run citation classification",
        description="Run citation classification with the provided arguments.",
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
        choices=_CONTEXT_USER_PROMPTS.keys(),
        default="sentence",
        help="The user prompt to use for context classification. Defaults to"
        " %(default)s.",
    )
    run_parser.add_argument(
        "--ref-limit",
        type=int,
        default=None,
        help="The number of references per paper to process. Defaults to all.",
    )
    run_parser.add_argument(
        "--continue-papers",
        type=Path,
        default=None,
        help="Path to file with data from a previous run",
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
            classify_contexts(
                args.model,
                args.api_key,
                args.data_path,
                args.limit,
                args.user_prompt,
                args.output_dir,
                args.ref_limit,
                args.continue_papers,
            )
        )


def list_prompts(detail: bool) -> None:
    items = [
        ("CONTEXT PROMPTS", _CONTEXT_USER_PROMPTS),
    ]
    for title, prompts in items:
        print()
        if detail:
            print(">>>", title)
        else:
            print(title)
        for key, prompt in prompts.items():
            if detail:
                sep = "-" * 80
                print(f"{sep}\n{key}\n{sep}\n{prompt}")
            else:
                print(f"- {key}")


if __name__ == "__main__":
    main()
