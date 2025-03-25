"""Extract Atomic Content Units (ACUs) from related paper abstracts.

The input can be the either the main or related papers from `paper construct`:
- peerread_with_s2_references.json (s2.PaperWithS2Refs): `peerread` type
- peerread_related.json (s2.Paper): `s2` type

Select the appropriate type with `--paper-type`.

User prompt and output type definition from Ai et al 2025, p. 11.
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Self

import dotenv
import typer
from pydantic import BaseModel, ConfigDict, Field

from paper.gpt.model import (
    PaperACUInput,
    PaperACUType,
    PaperWithACUs,
    Prompt,
    PromptResult,
)
from paper.gpt.prompts import PromptTemplate, load_prompts, print_prompts
from paper.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    ModelClient,
    append_intermediate_result,
    init_remaining_items,
)
from paper.util import (
    Timer,
    cli,
    describe,
    ensure_envvar,
    get_params,
    progress,
    render_params,
    sample,
    setup_logging,
)
from paper.util.serde import load_data, save_data

logger = logging.getLogger(__name__)

ACU_EXTRACTION_USER_PROMPTS = load_prompts("extract_acu")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def run(
    peerread_path: Annotated[
        Path,
        typer.Option(
            "--input", help="The path to the JSON file containing the input papers."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            help="The path to the output directory where the files will be saved.",
        ),
    ],
    paper_type: Annotated[
        PaperACUType, typer.Option(help="Type of paper for the input data.")
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="The model to use for the extraction."),
    ] = "gpt-4o-mini",
    limit_papers: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="The number of papers to process. Use 0 for all papers.",
        ),
    ] = 10,
    user_prompt: Annotated[
        str,
        typer.Option(
            help="The user prompt to use for ACU extraction.",
            click_type=cli.Choice(ACU_EXTRACTION_USER_PROMPTS),
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
        bool, typer.Option(help="Keep intermediate results.")
    ] = False,
    seed: Annotated[
        int,
        typer.Option(help="Random seed used for the GPT API and to shuffle the data."),
    ] = 0,
) -> None:
    """Evaluate each review's novelty rating based on the review text."""
    asyncio.run(
        extract_acu(
            model,
            peerread_path,
            paper_type,
            limit_papers,
            user_prompt,
            output_dir,
            continue_papers,
            continue_,
            seed,
            keep_intermediate,
        )
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()


_EXTRACT_ACU_SYSTEM_PROMPT = (
    "You are an expert at extracting atomic content units (ACUs) from text. Given a"
    " paper abstract, extract the list of ACUs."
)


async def extract_acu(
    model: str,
    related_path: Path,
    paper_type: PaperACUType,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    continue_papers_file: Path | None,
    continue_: bool,
    seed: int,
    keep_intermediate: bool,
) -> None:
    """Extract ACUs from each related paper's abstract.

    Args:
        model: GPT model code to use.
        related_path: Path to the JSON file containing the input papers data.
        paper_type: Type of the input papers.
        limit_papers: Number of papers to process. If 0 or None, process all.
        user_prompt_key: Key to the user prompt to use for paper evaluation. See
            `EXTRACT_ACU_SYSTEM_PROMPT` for available options or `list_prompts` for
            more.
        output_dir: Directory to save the output files.
        continue_papers_file: If provided, check for entries in the input data.
        continue_: If True, use data from `continue_papers_file`.
        keep_intermediate: Keep intermediate results to be used with `continue`.
        seed: Seed for the OpenAI API call and to shuffle the data.
    """
    random.seed(seed)
    params = get_params()
    logger.info(render_params(params))

    dotenv.load_dotenv()

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    if limit_papers == 0:
        limit_papers = None

    client = ModelClient(
        api_key=ensure_envvar("OPENAI_API_KEY"), model=model, seed=seed
    )

    paper_type_ = paper_type.get_type()
    papers = sample(load_data(related_path, paper_type.get_type()), limit_papers)
    user_prompt = ACU_EXTRACTION_USER_PROMPTS[user_prompt_key]

    output_intermediate_file, papers_remaining = init_remaining_items(
        PaperWithACUs[paper_type_],
        output_dir,
        continue_papers_file,
        papers,
        continue_,
    )

    with Timer() as timer:
        results = await _extract_acus(
            client,
            user_prompt,
            paper_type_,
            papers_remaining.remaining,
            output_intermediate_file,
            keep_intermediate,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    results_all = papers_remaining.done + results.result
    results_items = PromptResult.unwrap(results_all)

    acus_total = sum(len(i.acus) for i in results_items)
    salient_total = sum(len(i.salient_acus) for i in results_items)
    logger.info("Papers: %d.", len(results_items))
    logger.info(
        "Total ACUs: %d. Average: %f.", acus_total, acus_total / len(results_items)
    )
    logger.info(
        "Total salient ACUs: %d. Average: %f.",
        salient_total,
        salient_total / len(results_items),
    )
    acu_word_lengths = [len(acu.split()) for i in results_items for acu in i.acus]
    logger.info("Words per ACUs:\n%s", describe(acu_word_lengths))

    save_data(output_dir / "result.json", results_all)
    save_data(output_dir / "params.json", params)

    if len(results_all) != len(papers):
        logger.warning("Some papers are missing from the result.")


async def _extract_acus[T: PaperACUInput](
    client: ModelClient,
    user_prompt: PromptTemplate,
    type_: type[T],
    papers: Sequence[T],
    output_intermediate_file: Path,
    keep_intermediate: bool,
) -> GPTResult[list[PromptResult[PaperWithACUs[T]]]]:
    """Extract ACUs for each related paper's abstract.

    Args:
        client: OpenAI client to use GPT.
        user_prompt: User prompt template to use for extraction to be filled.
        type_: Type of the input paper.
        papers: Related papers from the S2 API.
        output_intermediate_file: File to write new results after each task.
        keep_intermediate: Keep intermediate results to be used in future runs.

    Returns:
        List of papers with evaluated reviews wrapped in a GPTResult.
    """
    results: list[PromptResult[PaperWithACUs[T]]] = []
    total_cost = 0

    tasks = [_extract_acu_single(client, type_, paper, user_prompt) for paper in papers]

    for task in progress.as_completed(tasks, desc="Processing papers"):
        result = await task
        total_cost += result.cost

        results.append(result.result)
        if keep_intermediate:
            append_intermediate_result(output_intermediate_file, result.result)

    return GPTResult(results, total_cost)


class GPTACU(BaseModel):
    """Structured output for ACU extraction from paper abstract."""

    model_config = ConfigDict(frozen=True)

    summary: Annotated[str, Field(description="Document summary.")]
    all_acus: Annotated[list[str], Field(description="Array of ACU strings.")]
    salient_acus: Annotated[
        list[str], Field(description="Array of salient ACU strings.")
    ]

    @classmethod
    def empty(cls) -> Self:
        """Create empty ACU result with empty ACU lists and summary "<error>"."""
        return cls(summary="<error>", all_acus=[], salient_acus=[])

    def is_empty(self) -> bool:
        """Check if extraction is invalid: summary is "<error>"."""
        return self.summary == "<error>"


async def _extract_acu_single[T: PaperACUInput](
    client: ModelClient, type_: type[T], paper: T, user_prompt: PromptTemplate
) -> GPTResult[PromptResult[PaperWithACUs[T]]]:
    """Extract ACUs for a single paper.

    Args:
        client: OpenAI client to use GPT.
        type_: Type of the input paper.
        paper: Input paper (S2 related paper or PeerRead main paper).
        user_prompt: User prompt template to use for extraction to be filled.

    Returns:
        S2 paper with extracted ACUs wrapped in a GPTResult.
    """
    user_prompt_text = user_prompt.template.format(
        title=paper.title, abstract=paper.abstract
    )
    result = await client.run(GPTACU, _EXTRACT_ACU_SYSTEM_PROMPT, user_prompt_text)
    item = result.result or GPTACU.empty()

    if item.is_empty():
        logger.warning(f"Paper '{paper.title}': invalid ACUs")

    return GPTResult(
        result=PromptResult(
            item=PaperWithACUs[type_].from_(
                paper, item.all_acus, item.salient_acus, item.summary
            ),
            prompt=Prompt(system=_EXTRACT_ACU_SYSTEM_PROMPT, user=user_prompt_text),
        ),
        cost=result.cost,
    )


@app.command(help="List available prompts.")
def prompts(
    detail: Annotated[
        bool, typer.Option(help="Show full description of the prompts.")
    ] = False,
) -> None:
    """Print the available prompt names, and optionally, the full prompt text."""
    print_prompts("ACU EXTRACTION", ACU_EXTRACTION_USER_PROMPTS, detail=detail)


if __name__ == "__main__":
    app()
