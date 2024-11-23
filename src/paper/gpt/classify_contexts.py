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

import asyncio
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from paper import evaluation_metrics
from paper.asap.model import (
    ContextPolarityBinary,
    PaperReview,
    PaperSection,
    PaperWithFullReference,
)
from paper.asap.model import PaperWithFullReference as PaperInput
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
    Timer,
    cli,
    display_params,
    ensure_envvar,
    progress,
    safediv,
    setup_logging,
)
from paper.util.serde import Record

logger = logging.getLogger(__name__)


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


class PaperOutput(Record):
    """Paper where its references have contexts with polarity."""

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    reviews: Sequence[PaperReview] = Field(description="Feedback from a reviewer")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[Reference] = Field(description="References made in the paper")

    @property
    def id(self) -> int:
        return hash(self.title + self.abstract)


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
    *,
    seed: int,
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
                seed=seed,
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
            reviews=paper.reviews,
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
    *,
    seed: int,
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
        _classify_paper(client, limit_references, model, paper, user_prompt, seed=seed)
        for paper in papers
    ]
    for task in progress.as_completed(
        tasks, desc="Classifying paper reference contexts"
    ):
        result = await task
        total_cost += result.cost

        paper_outputs.append(result.result)
        append_intermediate_result(PaperOutput, output_intermediate_path, result.result)

    return GPTResult(paper_outputs, total_cost)


async def classify_contexts(
    model: str,
    data_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    limit_references: int | None,
    continue_papers_file: Path | None,
    clean_run: bool,
    seed: int,
) -> None:
    """Classify reference citation contexts by polarity."""
    logger.info(display_params())

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    if limit_references == 0:
        limit_references = None

    client = AsyncOpenAI(api_key=ensure_envvar("OPENAI_API_KEY"))

    data = TypeAdapter(list[PaperInput]).validate_json(data_path.read_bytes())

    papers = data[:limit_papers]
    user_prompt = _CONTEXT_USER_PROMPTS[user_prompt_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperOutput, output_intermediate_file, continue_papers_file, papers, clean_run
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

    with Timer() as timer:
        results = await _classify_contexts(
            client,
            model,
            user_prompt,
            papers_remaining.remaining,
            limit_references,
            output_intermediate_file,
            seed=seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    stats, metrics = show_classified_stats(result.item for result in results_all)
    logger.info("Classification metrics:\n%s\n", stats)

    assert len(results_all) == len(papers)
    (output_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperOutput]]).dump_json(results_all, indent=2)
    )
    (output_dir / "output.txt").write_text(stats)
    if metrics is not None:
        (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))


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


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    data_path: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the papers data."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The path to the output directory where files will be saved."
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="The model to use for the extraction.",
            click_type=cli.choice(MODELS_ALLOWED),
        ),
    ] = "gpt-4o-mini",
    limit_papers: Annotated[
        int,
        typer.Option("--limit", "-n", help="The number of papers to process."),
    ] = 1,
    user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for context classification.",
            click_type=cli.choice(_CONTEXT_USER_PROMPTS),
        ),
    ] = "sentence",
    limit_references: Annotated[
        int | None,
        typer.Option(
            "--ref-limit",
            help="Limit to the number of references per paper to process.",
        ),
    ] = None,
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run")
    ] = None,
    clean_run: Annotated[
        bool,
        typer.Option(help="Start from scratch, ignoring existing intermediate results"),
    ] = False,
    seed: Annotated[int, typer.Option(help="Seed to set in the OpenAI call.")] = 0,
) -> None:
    asyncio.run(
        classify_contexts(
            model,
            data_path,
            limit_papers,
            user_prompt,
            output_dir,
            limit_references,
            continue_papers,
            clean_run,
            seed,
        )
    )


@app.callback()
def main() -> None:
    setup_logging()


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    print_prompts("CONTEXT PROMPTS", _CONTEXT_USER_PROMPTS, detail=detail)


if __name__ == "__main__":
    app()
