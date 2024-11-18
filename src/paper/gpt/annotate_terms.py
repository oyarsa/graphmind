"""Extract key terms for problems and methods from S2 Papers.

Input is a JSON array of `paper.external_data.semantic_scholar.model.Paper`. Output
contains the input paper plus the prompts used and the extracted terms.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean, stdev
from typing import Annotated, Any, Self, override

import dotenv
from openai import AsyncClient, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.console import Console
from rich.table import Table

from paper import scimon
from paper.gpt.model import Prompt, PromptResult, S2Paper
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
    Timer,
    display_params,
    mustenv,
    progress,
    safediv,
    setup_logging,
)
from paper.util.serde import Record, load_data, save_data

logger = logging.getLogger(__name__)


_TERM_SYSTEM_PROMPT = """\
You are a helpful assistant that can read scientific papers and identify the key terms \
used to describe the problems tackled by the paper and the terms used for the methods.
"""
_TERM_USER_PROMPTS = load_prompts("annotate_terms")

_SPLIT_SYSTEM_PROMPT = """\
You are a helpful assistant that can read a paper abstract and identify which sentences \
describe the paper background context, and which describe the paper goals and target.
"""
_SPLIT_USER_PROMPTS = load_prompts("split_abstract")


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
                args.user_prompt_term,
                args.user_prompt_split,
                args.continue_papers,
                args.clean_run,
                args.log,
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
        "--user-prompt-term",
        type=str,
        choices=sorted(_TERM_USER_PROMPTS),
        default="multi",
        help="The user prompt to use for term annotation.",
    )
    run_parser.add_argument(
        "--user-prompt-split",
        type=str,
        choices=sorted(_SPLIT_USER_PROMPTS),
        default="simple",
        help="The user prompt to use for abstract splitting.",
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
        "--log",
        default=DetailOptions.NONE,
        type=DetailOptions,
        choices=list(DetailOptions),
        help="How much detail to show in output logging",
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
    user_prompt_term_key: str,
    user_prompt_split_key: str,
    continue_papers_file: Path | None,
    clean_run: bool,
    show_log: DetailOptions,
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
        user_prompt_term_key: Key of the prompt to use in the annotation prompt mapping.
        user_prompt_split_key: Key of the prompt to use in the abstract splitting.
        continue_papers_file: File with the intermediate results from a previous run
            that we want to continue.
        clean_run: If True, we ignore `continue_papers_file` and start from scratch.
        show_log: Show log of term count for each paper and type of term. If NONE, show
            only statistics on entity validation. If TABLE, shows a summary table of
            each term type. If DETAIL, shows detailed information on the extracted
            entities. Note: the types of terms vary by output type, dependent on the
            prompt.
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
    user_prompt_terms = _TERM_USER_PROMPTS[user_prompt_term_key]
    term_type = _TERM_TYPES[user_prompt_terms.type_name]
    user_prompt_split = _SPLIT_USER_PROMPTS[user_prompt_split_key]

    papers = load_data(input_file, S2Paper)[:limit_papers]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_intermediate_file = output_dir / "results.tmp.json"
    papers_remaining = get_remaining_items(
        PaperAnnotatedTerms[term_type],
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

    with Timer() as timer:
        output = await _annotate_papers_terms(
            client,
            model,
            papers_remaining.remaining,
            user_prompt_terms,
            user_prompt_split,
            output_intermediate_file,
            term_type,
            seed=seed,
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${output.cost:.10f}")

    save_data(output_dir / "results.json", output.result)
    assert len(papers) == len(output.result)

    if show_log:
        _log_table_stats(output.result, detail=show_log)


class GPTAbstractSplit(BaseModel):
    """Describes the division of the paper abstract in two groups of sentences."""

    model_config = ConfigDict(frozen=True)

    context: Annotated[
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


@dataclass(frozen=True, kw_only=True)
class EntityValidation:
    violations: Sequence[str]
    entities: int
    entities_invalid: int

    @property
    def valid(self) -> bool:
        return self.entities_invalid == 0


class GPTTermBase(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @classmethod
    @abstractmethod
    def empty(cls) -> Self: ...

    @abstractmethod
    def validate_entities(self, text: str) -> EntityValidation:
        """Validate if all extracted entities are exact substrings of the `text`.

        Args:
            text: Text from where the entities were extracted.

        Returns:
            Validation result with the message, the number of entities and
            valid entities.
        """
        ...

    @abstractmethod
    def to_scimon(self) -> scimon.Terms: ...


class GPTTermRelation(BaseModel):
    """Represents a directed relation between two scientific terms.

    Relations are head --type-> tail.
    """

    head: Annotated[str, Field(description="Head term of the relation.")]
    tail: Annotated[str, Field(description="Tail term of the relation.")]


class GPTMultiTerms(GPTTermBase):
    """Structured output for scientific term extraction."""

    tasks: Annotated[
        Sequence[str],
        Field(description="Core problems, objectives or applications addressed."),
    ]
    methods: Annotated[
        Sequence[str],
        Field(
            description="Technical approaches, algorithms, or frameworks used/proposed."
        ),
    ]
    metrics: Annotated[
        Sequence[str], Field(description="Evaluation metrics and measures mentioned.")
    ]
    resources: Annotated[
        Sequence[str], Field(description="Datasets, resources, or tools utilised.")
    ]
    relations: Annotated[
        Sequence[GPTTermRelation],
        Field(description="Directed relations between terms."),
    ]

    @override
    @classmethod
    def empty(cls) -> Self:
        return cls(tasks=[], methods=[], metrics=[], resources=[], relations=[])

    @override
    def validate_entities(self, text: str) -> EntityValidation:
        if not text:
            return EntityValidation(
                violations=["Empty text."], entities=0, entities_invalid=0
            )

        text = text.casefold()
        violations: list[str] = []

        entities = list(
            itertools.chain(self.tasks, self.methods, self.metrics, self.resources)
        )
        for entity in entities:
            if entity.casefold() not in text:
                violations.append(f"Entity: '{entity}'.")

        for relation in self.relations:
            if relation.head.casefold() not in text:
                violations.append(f"Head: '{relation.head}'")
            if relation.tail.casefold() not in text:
                violations.append(f"Tail: '{relation.tail}'")

        entities_num = len(entities) + len(self.relations) * 2
        return EntityValidation(
            violations=violations,
            entities=entities_num,
            entities_invalid=len(violations),
        )

    @override
    def to_scimon(self) -> scimon.Terms:
        return scimon.Terms(
            tasks=self.tasks,
            methods=self.methods,
            metrics=self.metrics,
            resources=self.resources,
            relations=[
                scimon.Relation(head=relation.head, tail=relation.tail)
                for relation in self.relations
            ],
        )


_TERM_TYPES: Mapping[str, type[GPTTermBase]] = {
    "multi-terms": GPTMultiTerms,
}


class PaperAnnotatedTerms[T: GPTTermBase](Record):
    """S2 Paper with its annotated key terms. Includes GPT prompts used."""

    terms: T
    paper: S2Paper
    context: str
    target: str

    @property
    def id(self) -> str:
        return self.paper.id

    @computed_field
    @property
    def valid(self) -> EntityValidation:
        return self.terms.validate_entities(self.paper.abstract or "")

    def to_scimon(self) -> scimon.Paper:
        abstract = self.paper.abstract or "<no abstract>"
        return scimon.Paper(
            id=self.id,
            terms=self.terms.to_scimon(),
            abstract=abstract,
            context=self.context,
            target=self.target,
        )


async def _annotate_papers_terms[T: GPTTermBase](
    client: AsyncClient,
    model: str,
    papers: Sequence[S2Paper],
    user_prompt_term: PromptTemplate,
    user_prompt_split: PromptTemplate,
    output_intermediate_path: Path,
    term_type: type[T],
    *,
    seed: int,
) -> GPTResult[list[PromptResult[PaperAnnotatedTerms[T]]]]:
    """Annotate papers to add key terms. Runs multiple tasks concurrently."""
    ann_outputs: list[PromptResult[PaperAnnotatedTerms[T]]] = []
    total_cost = 0

    tasks = [
        _annotate_paper_term_single(
            client, model, seed, paper, user_prompt_term, user_prompt_split, term_type
        )
        for paper in papers
    ]

    for task in progress.as_completed(tasks, desc="Extracting paper terms"):
        result = await task
        total_cost += result.cost

        ann_outputs.append(result.result)
        append_intermediate_result(
            PaperAnnotatedTerms[term_type], output_intermediate_path, result.result
        )

    return GPTResult(result=ann_outputs, cost=total_cost)


async def _annotate_paper_term_single[T: GPTTermBase](
    client: AsyncClient,
    model: str,
    seed: int,
    paper: S2Paper,
    user_prompt_term: PromptTemplate,
    user_prompt_split: PromptTemplate,
    term_type: type[T],
) -> GPTResult[PromptResult[PaperAnnotatedTerms[T]]]:
    """Annotate a single paper with its key terms."""
    split_prompt_text = user_prompt_term.template.format(
        title=paper.title, abstract=paper.abstract
    )
    result_term = await run_gpt(
        term_type,
        client,
        _TERM_SYSTEM_PROMPT,
        split_prompt_text,
        model,
        seed=seed,
    )
    terms = result_term.result if result_term.result else term_type.empty()

    split_prompt_text = user_prompt_split.template.format(abstract=paper.abstract)
    result_split = await run_gpt(
        GPTAbstractSplit,
        client,
        _SPLIT_SYSTEM_PROMPT,
        split_prompt_text,
        model,
        seed=seed,
    )
    split = (
        result_split.result
        if result_split.result
        else GPTAbstractSplit(context="", target="")
    )

    return GPTResult(
        result=PromptResult(
            item=PaperAnnotatedTerms(
                terms=terms,
                paper=paper,
                context=split.context,
                target=split.target,
            ),
            prompt=Prompt(user=split_prompt_text, system=_TERM_SYSTEM_PROMPT),
        ),
        cost=result_term.cost,
    )


class DetailOptions(StrEnum):
    NONE = "none"
    TABLE = "table"
    DETAIL = "detail"


def _log_table_stats(
    results: Sequence[PromptResult[PaperAnnotatedTerms[GPTTermBase]]],
    detail: DetailOptions,
) -> None:
    if not results:
        logger.warning("Cannot log stats from empty results.")
        return
    try:
        columns = results[0].item.terms.model_dump().keys() - {"relations"}
    except Exception:
        logger.exception("Could not get the term keys")
        return

    table = Table("paper", *columns, "validation")

    valid_full_count = 0
    invalid_count = 0
    entities_all = 0
    for result in results:
        terms = result.item.terms.model_dump()
        valid_result = result.item.terms.validate_entities(
            result.item.paper.abstract or ""
        )
        invalid_count += valid_result.entities_invalid
        entities_all += valid_result.entities

        e_all = valid_result.entities
        e_inv = valid_result.entities_invalid
        valid_msg = f"{e_inv}/{e_all} ({safediv(e_inv, e_all):.2%})"

        row = [
            result.item.paper.title,
            *(_format_col(terms[col], detail) for col in columns),
            _format_valid_msg(valid_msg, valid_result.valid),
        ]
        valid_full_count += valid_result.entities_invalid == 0
        table.add_row(*map(str, row))

    if detail is not DetailOptions.NONE:
        console = Console()
        with console.capture() as capture:
            console.print(table)
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

    logger.info(
        f"Full valid: {valid_full_count}/{len(results)}"
        f" ({safediv(valid_full_count, len(results)):.2%})"
    )
    logger.info(
        f"Invalid overall: {invalid_count}/{entities_all}"
        f" ({safediv(invalid_count, entities_all):.2%})"
    )
    logger.info("Term counts per paper (mean ± stdev):")
    for col in term_averages:
        logger.info(f"  {col}: {term_averages[col]:.2f} ± {term_stdevs[col]:.2f}")


def _format_col(col: Sequence[Any], detail: DetailOptions) -> str:
    if detail is DetailOptions.DETAIL:
        return "\n".join(f"{i}. {c}" for i, c in enumerate(col, 1)) or "-"
    return str(len(col))


def _format_valid_msg(msg: str, valid: bool) -> str:
    colour = "green" if valid else "red"
    return f"[{colour}]{msg}"


def list_prompts(detail: bool) -> None:
    print_prompts("TERM ANNOTATION PROMPTS", _TERM_USER_PROMPTS, detail=detail)
