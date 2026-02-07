"""Extract key terms for problems and methods from S2 Papers.

Input is a JSON array of `paper.semantic_scholar.model.Paper`. Output contains the input
paper plus the prompts used and the extracted terms.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from statistics import mean, stdev
from typing import Annotated, Any, Self

import typer
from pydantic import Field
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from paper.gpt.model import (
    PaperAnnotated,
    PaperTerms,
    PaperToAnnotate,
    PaperType,
    Prompt,
    PromptResult,
)
from paper.gpt.prompts import PromptTemplate, print_prompts
from paper.gpt.prompts.abstract_classification import ABS_USER_PROMPTS
from paper.gpt.prompts.abstract_demonstrations import ABS_DEMO_PROMPTS
from paper.gpt.prompts.annotate_terms import TERM_USER_PROMPTS
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    LLMClient,
    append_intermediate_result,
    get_remaining_items,
)
from paper.types import Immutable
from paper.util import (
    Timer,
    cli,
    dotenv,
    get_params,
    progress,
    render_params,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)


TERM_SYSTEM_PROMPT = """\
You are a helpful assistant that can read scientific papers and identify the key terms \
used to describe the problems tackled by the paper and the terms used for the methods.
"""

ABS_SYSTEM_PROMPT = """\
You are a helpful assistant that can read a paper abstract and identify which sentences \
describe the paper background context, and which describe the paper goals and target.
"""


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


class DetailOptions(StrEnum):
    """Detail level for output information.

    Attributes:
        NONE: doesn't print a table.
        TABLE: prints table with summary of paper counts.
        DETAIL: prints table with full extracted papers.
    """

    NONE = "none"
    TABLE = "table"
    DETAIL = "detail"


@app.command(help=__doc__, no_args_is_help=True)
def run(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the JSON file containing the papers data (S2Paper format)."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The path to the output directory where files will be saved."
        ),
    ],
    paper_type: Annotated[
        PaperType, typer.Option(help="Type of paper for the input data")
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="The number of papers to process. Set to 0 for all papers.",
        ),
    ] = 0,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="The model to use for the annotation.",
            click_type=cli.Choice(MODELS_ALLOWED),
        ),
    ] = "gpt-4o-mini",
    seed: Annotated[int, typer.Option(help="Seed to set in the OpenAI call.")] = 0,
    prompt_term: Annotated[
        str,
        typer.Option(
            help="User prompt to use for term annotation.",
            click_type=cli.Choice(TERM_USER_PROMPTS),
        ),
    ] = "multi",
    prompt_abstract: Annotated[
        str,
        typer.Option(
            help="User prompt to use for abstract classification.",
            click_type=cli.Choice(ABS_USER_PROMPTS),
        ),
    ] = "simple",
    abstract_demonstrations: Annotated[
        Path | None,
        typer.Option(
            "--abstract-demos",
            help="Path to demonstrations containing data for abstract classification.",
        ),
    ] = None,
    abstract_demonstrations_prompt: Annotated[
        str,
        typer.Option(
            "--abstract-demo-prompt",
            help="Prompt used to create abstract classification demonstrations",
            click_type=cli.Choice(ABS_DEMO_PROMPTS),
        ),
    ] = "simple",
    continue_papers: Annotated[
        Path | None, typer.Option(help="Path to file with data from a previous run.")
    ] = None,
    continue_: Annotated[
        bool,
        typer.Option("--continue", help="Use existing intermediate results."),
    ] = False,
    log: Annotated[
        DetailOptions, typer.Option(help="How much detail to show in output logging.")
    ] = DetailOptions.NONE,
    batch_size: Annotated[
        int, typer.Option(help="Size of the batches being annotated.")
    ] = 100,
) -> None:
    """Extract key terms for problems and methods from S2 Papers."""
    asyncio.run(
        annotate_papers(
            input_file,
            output_dir,
            limit,
            model,
            seed,
            prompt_term,
            prompt_abstract,
            abstract_demonstrations,
            abstract_demonstrations_prompt,
            continue_papers,
            continue_,
            log,
            paper_type,
            batch_size,
        )
    )


async def annotate_papers(
    input_file: Path,
    output_dir: Path,
    limit_papers: int | None,
    model: str,
    seed: int,
    user_prompt_term_key: str,
    user_prompt_abstract_key: str,
    abstract_demonstrations_path: Path | None,
    abstract_demonstrations_prompt: str,
    continue_papers_file: Path | None,
    continue_: bool,
    show_log: DetailOptions,
    paper_type: PaperType,
    batch_size: int,
) -> None:
    """Extract problem and method terms from each paper.

    Papers whose extractions are invalid - terms are invalid, or empty background or
    target - are discarded.

    Args:
        input_file: JSON with input data.
            Array of `paper.semantic_scholar.model.Paper`.
        output_dir: Directory to save output files, including final and intermedaite
            results.
        limit_papers: How many papers to process. 0 or None means all papers.
        model: Name of the OpenAI API model to use.
            See `papers.gpt.run_gpt.MODELS_ALLOWED` for the allowed ones.
        seed: Random generator seed to pass to the model to try to get some
            reproducibility.
        user_prompt_term_key: Key of the prompt to use in the annotation prompt mapping.
        user_prompt_abstract_key: Key of the prompt to use in the abstract
            classification.
        abstract_demonstrations_path: If present, path to data that will be used to
            construct abstract classification demonstrations.
        abstract_demonstrations_prompt: Prompt to build the demonstrations string.
        continue_papers_file: File with the intermediate results from a previous run
            that we want to continue.
        continue_: If True, we use data from `continue_papers_file`.
        show_log: Show log of term count for each paper and type of term. If NONE, show
            only statistics on the quantities of extracted terms. If TABLE, shows a
            summary table of each term type. If DETAIL, shows detailed information on
            the extracted entities. Note: the types of terms vary by output type,
            dependent on the prompt.
        paper_type: Type of the paper input data.
        batch_size: Number of items per annotation batch.
    """
    params = get_params()
    logger.info(render_params(params))

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = LLMClient.new_env(model=model, seed=seed)
    user_prompt_terms = TERM_USER_PROMPTS[user_prompt_term_key]
    user_prompt_abstract = ABS_USER_PROMPTS[user_prompt_abstract_key]

    papers = load_data(input_file, paper_type.get_type())[:limit_papers]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperAnnotated,
        output_intermediate_file,
        continue_papers_file,
        papers,
        continue_,
    )
    if not papers_remaining.remaining:
        logger.info(
            "No items left to process. They're all on the `continues` file. Exiting."
        )
        return

    if continue_:
        logger.info(
            "Skipping %d items from the `continue` file.", len(papers_remaining.done)
        )

    abstract_demonstrations = format_abstract_demonstrations(
        load_data(abstract_demonstrations_path, AbstractDemonstration)
        if abstract_demonstrations_path
        else [],
        ABS_DEMO_PROMPTS[abstract_demonstrations_prompt],
    )

    with Timer() as timer:
        output = await _annotate_papers(
            client,
            papers_remaining.remaining,
            user_prompt_terms,
            user_prompt_abstract,
            abstract_demonstrations,
            output_intermediate_file,
            batch_size,
        )
    output_valid = [ann for ann in output.result if ann.item.is_valid()]

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${output.cost:.10f}")

    logger.info("All results: %d", len(output.result))
    logger.info("Valid results: %d", len(output_valid))

    save_data(output_dir / "results_all.json.zst", output.result)
    save_data(output_dir / "results_valid.json.zst", output_valid)
    save_data(output_dir / "params.json", params)
    assert len(papers) == len(output.result)

    if show_log:
        _log_table_stats(output_valid, detail=show_log)


class GPTAbstractClassify(Immutable):
    """Describes the division of the paper abstract in two groups of sentences."""

    background: Annotated[
        str,
        Field(
            description="Sentences that describe the background context from the paper,"
            " for example the tasks and goals."
        ),
    ]
    target: Annotated[
        str,
        Field(
            description="Sentences that describe the target of the paper, including"
            " methods, resources, etc."
        ),
    ]

    @classmethod
    def empty(cls) -> Self:
        """Instance with empty strings for background and target."""
        return cls(background="", target="")

    def is_valid(self) -> bool:
        """Check if instance is valid, i.e. background and target are non-empty."""
        return bool(self.background and self.target)


async def _annotate_papers(
    client: LLMClient,
    papers: Sequence[PaperToAnnotate],
    user_prompt_term: PromptTemplate,
    user_prompt_abstract: PromptTemplate,
    abstract_demonstrations: str,
    output_intermediate_path: Path,
    batch_size: int,
) -> GPTResult[list[PromptResult[PaperAnnotated]]]:
    """Annotate papers to add key terms. Runs multiple tasks concurrently."""
    ann_outputs: list[PromptResult[PaperAnnotated]] = []
    total_cost = 0

    batches = list(itertools.batched(papers, batch_size))
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches"), 1):
        batch_tasks = [
            _annotate_paper_single(
                client,
                paper,
                user_prompt_term,
                user_prompt_abstract,
                abstract_demonstrations,
            )
            for paper in batch
        ]

        for task in progress.as_completed(
            batch_tasks, desc=f"Annotating batch {batch_idx}"
        ):
            result = await task
            total_cost += result.cost

            ann_outputs.append(result.result)
            append_intermediate_result(output_intermediate_path, result.result)

    return GPTResult(result=ann_outputs, cost=total_cost)


async def _annotate_paper_single(
    client: LLMClient,
    paper: PaperToAnnotate,
    user_prompt_term: PromptTemplate,
    user_prompt_abstract: PromptTemplate,
    abstract_demonstrations: str,
) -> GPTResult[PromptResult[PaperAnnotated[PaperTerms]]]:
    """Annotate a single paper with its key terms."""
    term_prompt_text = user_prompt_term.template.format(
        title=paper.title, abstract=paper.abstract
    )
    abstract_prompt_text = user_prompt_abstract.template.format(
        demonstrations=abstract_demonstrations, abstract=paper.abstract
    )

    result_term = await client.run(PaperTerms, TERM_SYSTEM_PROMPT, term_prompt_text)
    result_abstract = await client.run(
        GPTAbstractClassify, ABS_SYSTEM_PROMPT, abstract_prompt_text
    )

    terms = result_term.result or PaperTerms.empty()
    abstract = result_abstract.result or GPTAbstractClassify.empty()

    if not terms.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid PaperTerms")
    if not abstract.is_valid():
        logger.warning(f"Paper '{paper.title}': invalid GPTAbstractClassify")

    return GPTResult(
        result=PromptResult(
            item=PaperAnnotated(
                terms=terms,
                paper=paper,
                background=abstract.background,
                target=abstract.target,
            ),
            prompt=Prompt(user=abstract_prompt_text, system=ABS_SYSTEM_PROMPT),
        ),
        cost=result_term.cost + result_abstract.cost,
    )


def _log_table_stats(
    results: Sequence[PromptResult[PaperAnnotated[PaperTerms]]],
    detail: DetailOptions,
) -> None:
    """Print table summarising the results.

    See `DetailOptions` for how the results are presented.
    """
    if not results:
        logger.warning("Cannot log stats from empty results.")
        return
    try:
        columns = results[0].item.terms.model_dump().keys() - {"relations"}
    except Exception:
        logger.exception("Could not get the term keys")
        return

    console = Console()

    if detail is not DetailOptions.NONE:
        table_papers = Table("paper", *columns, title="Paper results")

        for result in results:
            terms = result.item.terms.model_dump()

            row = [
                result.item.paper.title,
                *(_format_col(terms[col], detail) for col in columns),
            ]
            table_papers.add_row(*map(str, row))

        with console.capture() as capture:
            console.print(table_papers)
        logger.info("\n%s\n", capture.get())

    term_lengths = {
        col: [
            len(terms[col])
            for result in results
            if (terms := result.item.terms.model_dump())
        ]
        for col in results[0].item.terms.model_dump()
    }
    term_averages = {col: mean(lengths) for col, lengths in term_lengths.items()}
    term_stdevs = {
        col: stdev(lengths) if len(lengths) > 1 else 0.0
        for col, lengths in term_lengths.items()
    }

    table_counts = Table("Term", "Sum", "Mean", "Stdev", title="Term stats")
    for col in term_averages:
        table_counts.add_row(
            col,
            str(sum(term_lengths[col])),
            f"{term_averages[col]:.2f}",
            f"{term_stdevs[col]:.2f}",
        )

    with console.capture() as capture:
        console.print(table_counts)
    logger.info("\n%s", capture.get())


def _format_col(col: Sequence[Any], detail: DetailOptions) -> str:
    """Format column for table output.

    By default, returns just the number of characters in the column.

    If `detail` is `DetailOptions`, instead returns all items as a numbered list, or `-`
    if `col` is empty.
    """
    if detail is DetailOptions.DETAIL:
        return "\n".join(f"{i}. {c}" for i, c in enumerate(col, 1)) or "-"
    return str(len(col))


class AbstractDemonstration(Immutable):
    """Prompt demonstration from an existing abstract extraction from CSAbstruct."""

    abstract: str
    background: str
    target: str


def format_abstract_demonstrations(
    data: Sequence[AbstractDemonstration], prompt: PromptTemplate
) -> str:
    """Format abstract demonstrations for prompt.

    Includes the demonstration instructions, followed by each demonstration formatted
    using `prompt`.
    """
    if not data:
        return ""

    output = [
        "-Demonstrations-",
        "",
        "The following are examples of abstracts and their correct classification into"
        " background and target.",
        "",
        "\n------\n".join(
            prompt.template.format(
                abstract=entry.abstract,
                background=entry.background,
                target=entry.target,
            )
            for entry in data
        ),
        "------",
    ]
    return "\n".join(output)


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    items = [
        ("TERM ANNOTATION PROMPTS", TERM_USER_PROMPTS),
        ("ABSTRACT CLASSIFICATION PROMPTS", ABS_USER_PROMPTS),
        ("ABSTRACT DEMONSTRATION PROMPTS", ABS_DEMO_PROMPTS),
    ]
    for title, prompts in items:
        print_prompts(title, prompts, detail=detail)
        print()
