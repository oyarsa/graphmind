"""Classify paper citation contexts by polarity and type using GPT-4.

Each paper contains many references. These can appear in one or more citation contexts
inside the text. Each citation context can be classified by polarity (positive vs
negative) and type (result, method, etc.)
"""

import argparse
import hashlib
import logging
import os
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from tqdm import tqdm

from paper_hypergraph.asap.model import PaperSection
from paper_hypergraph.asap.model import PaperWithFullReference as PaperInput
from paper_hypergraph.gpt.run_gpt import MODELS_ALLOWED, GptResult, run_gpt
from paper_hypergraph.util import BlockTimer, setup_logging

logger = logging.getLogger("classify_contexts")


class ContextPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ContextType(StrEnum):
    RESULT = "result"
    METHOD = "method"
    CONTRIBUTION = "contribution"
    OTHER = "other"


class ContextClassified(BaseModel):
    """Context from a paper reference with its classified polarity and type."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Full text of the context mention")
    polarity: ContextPolarity = Field(
        description="Whether the citation context is positive or negative"
    )
    type: ContextType = Field(description="Type of the context mention")


class Reference(BaseModel):
    """Paper reference where its contexts are enriched with polarity and type."""

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
    """Paper where its references have contexts with polarity and type."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Paper title")
    abstract: str = Field(description="Abstract text")
    ratings: Sequence[int] = Field(description="Reviewer ratings (1 to 5)")
    sections: Sequence[PaperSection] = Field(description="Sections in the paper text")
    approval: bool = Field(
        description="Approval decision - whether the paper was approved"
    )
    references: Sequence[Reference] = Field(description="References made in the paper")


_MODEL_SYNONYMS = {
    "4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "4o": "gpt-4o-2024-08-06",
    "gpt-4o": "gpt-4o-2024-08-06",
}


_CONTEXT_SYSTEM_PROMPT = (
    "Classify the context type and polarity between the main paper and its citation."
)
_CONTEXT_USER_PROMPTS = {
    "simple": """\
You are given a main paper and a reference with a citation context. Based on the main
paper's title and abstracts, the reference title and abstract, the citation context in
which the main paper mentions the reference, your task is to determine:

- The type of reference given by the citation context: one of 'result', 'method', \
'contribution' or 'other'. This represents the setting where the reference was made.
- The polarity: one of 'positive' or 'negative'. This represents whether the citation \
context is supporting the paper's goals (positive), or if it's provided as a \
counterpoint ('negative').

Main paper title: {main_title}
Main paper abstract:
{main_abstract}

Reference title: {reference_title}
Reference abstract:
{reference_abstract}

Citation context: {context}
"""
}


class GptContext(BaseModel):
    """Context from a paper reference with GPT-classified polarity and type.

    NB: This is currently identical to the main type (PaperContextClassified). They're
    separate on purpose.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Full text of the context mention")
    polarity: ContextPolarity = Field(
        description="Whether the citation context is positive or negative"
    )
    type: ContextType = Field(description="Type of the context mention")


def _classify_contexts(
    client: OpenAI, model: str, user_prompt_template: str, papers: Sequence[PaperInput]
) -> GptResult[list[PaperOutput]]:
    """Classify the contexts for each papers' references by polarity and type.

    Polarity (ContextPolarity): positive (supports argument) or negative (counterpoint).
    Types (ContextType): result, method, contribution, other.

    The returned object `PaperOutput` is very similar to the input `PaperInput`, but
    the references are different. Instead of the contexts being only strings, they
    are now `ContextClassified`, containing the original text plus the predicted
    polarity and type.
    """
    paper_outputs: list[PaperOutput] = []
    total_cost = 0

    for paper in tqdm(papers, desc="Classifying contexts"):
        classified_references: list[Reference] = []

        for reference in paper.references:
            classified_contexts: list[ContextClassified] = []

            for context in reference.contexts:
                user_prompt = user_prompt_template.format(
                    main_title=paper.title,
                    main_abstract=paper.abstract,
                    reference_title=reference.s2title,
                    reference_abstract=reference.abstract,
                    context=context,
                )
                result = run_gpt(
                    GptContext, client, _CONTEXT_SYSTEM_PROMPT, user_prompt, model
                )
                total_cost += result.cost

                if context := result.result:
                    classified_contexts.append(
                        ContextClassified(
                            text=context.text,
                            polarity=context.polarity,
                            type=context.type,
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
        assert len(classified_references) == len(paper.references)

        paper_outputs.append(
            PaperOutput(
                title=paper.title,
                abstract=paper.abstract,
                ratings=paper.ratings,
                sections=paper.sections,
                approval=paper.approval,
                references=classified_references,
            )
        )

    assert len(paper_outputs) == len(papers)
    return GptResult(paper_outputs, total_cost)


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit: int | None,
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
        f"  Limit: {limit if limit is not None else 'All'}\n"
        f"  User prompt: {user_prompt}\n"
    )


def classify_contexts(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit: int | None,
    user_prompt_key: str,
    output_dir: Path,
) -> None:
    """Classify reference citation contexts by polarity and type."""

    dotenv.load_dotenv()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = _MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    _log_config(
        model=model,
        data_path=data_path,
        limit=limit,
        user_prompt=user_prompt_key,
        output_dir=output_dir,
    )

    client = OpenAI()

    data = TypeAdapter(list[PaperInput]).validate_json(data_path.read_text())

    papers = data[:limit]
    user_prompt = _CONTEXT_USER_PROMPTS[user_prompt_key]

    with BlockTimer() as timer:
        results = _classify_contexts(client, model, user_prompt, papers)

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    classification_dir = output_dir / "context"
    (classification_dir / "result.json").write_text(
        TypeAdapter(list[PaperOutput]).dump_json(results.result, indent=2).decode()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for 'run' and 'prompts' subcommands
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        dest="subcommand",
        help="Additional help",
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
        help="The model to use for the extraction.",
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
        help="The number of papers to process. Defaults to 1 example.",
    )
    run_parser.add_argument(
        "--user-prompt",
        type=str,
        choices=_CONTEXT_USER_PROMPTS.keys(),
        default="bullets",
        help="The user prompt to use for the graph extraction. Defaults to %(default)s.",
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

    args = parser.parse_args()
    setup_logging(logger)

    if args.subcommand == "prompts":
        list_prompts(detail=args.detail)
    elif args.subcommand == "run":
        classify_contexts(
            args.model,
            args.api_key,
            args.data_path,
            args.limit,
            args.user_prompt,
            args.output_dir,
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
