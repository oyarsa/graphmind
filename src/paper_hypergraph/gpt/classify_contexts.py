"""Classify paper citation contexts by polarity using GPT-4.

Each paper contains many references. These can appear in one or more citation contexts
inside the text. Each citation context can be classified by polarity (positive vs
negative).
"""

import argparse
import hashlib
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from tqdm import tqdm

from paper_hypergraph import evaluation_metrics
from paper_hypergraph.asap.model import ContextPolarityBinary, PaperSection
from paper_hypergraph.asap.model import PaperWithFullReference as PaperInput
from paper_hypergraph.gpt.run_gpt import (
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GPTResult,
    Prompt,
    PromptResult,
    run_gpt,
)
from paper_hypergraph.util import Timer, setup_logging

logger = logging.getLogger("gpt.classify_contexts")


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
_CONTEXT_USER_PROMPTS = {
    "sentence": """\
You are given a citation context from a scientific paper that mentions. Your task is to \
determine the polarity of the citation context as 'positive' or 'negative'.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Citation context: {context}
#####
Output:
""",
    "simple": """\
You are given a main paper and a reference with a citation context. Your task is to \
determine the polarity of the citation context as 'positive' or 'negative', given the \
main paper's title, the reference's title, and the citation context where the main \
paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {main_title}
Reference title: {reference_title}
Citation context: {context}
#####
Output:
""",
    "full": """\
You are given a main paper and a reference with a citation context. Your task is to
determine the polarity of the citation context as 'positive' or 'negative', given the
main paper's title and abstract, the reference's title and abstract, and the citation
context where the main paper mentions the reference.

The polarity represents whether the citation context is supporting the paper's goals \
('positive'), or if it's provided as a counterpoint or criticism ('negative').

#####
-Data-
Main paper title: {main_title}
Main paper abstract: {main_abstract}

Reference title: {reference_title}
Reference abstract: {reference_abstract}

Citation context: {context}
#####
Output:
""",
}


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


def _classify_contexts(
    client: OpenAI,
    model: str,
    user_prompt_template: str,
    papers: Sequence[PaperInput],
    limit_references: int | None,
    use_expanded_context: bool,
) -> GPTResult[list[PromptResult[PaperOutput]]]:
    """Classify the contexts for each papers' references by polarity.

    Polarity (ContextPolarity): positive (supports argument) or negative (counterpoint).

    The returned object `PaperOutput` is very similar to the input `PaperInput`, but
    the references are different. Instead of the contexts being only strings, they
    are now `ContextClassified`, containing the original text plus the predicted
    polarity.
    """
    paper_outputs: list[PromptResult[PaperOutput]] = []
    user_prompts: list[str] = []
    total_cost = 0

    for paper in tqdm(papers, desc="Classifying contexts"):
        classified_references: list[Reference] = []

        references = paper.references[:limit_references]
        for reference in references:
            classified_contexts: list[ContextClassified] = []

            for context in reference.contexts_annotated_valid:
                context_text = (
                    context.expanded if use_expanded_context else context.regular
                )
                user_prompt = user_prompt_template.format(
                    main_title=paper.title,
                    main_abstract=paper.abstract,
                    reference_title=reference.s2title,
                    reference_abstract=reference.abstract,
                    context=context_text,
                )
                user_prompts.append(user_prompt)
                result = run_gpt(
                    GPTContext, client, _CONTEXT_SYSTEM_PROMPT, user_prompt, model
                )
                total_cost += result.cost

                if gpt_context := result.result:
                    classified_contexts.append(
                        ContextClassified(
                            text=context_text,
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
        assert len(classified_references) == len(references)

        paper_outputs.append(
            PromptResult(
                prompt=Prompt(
                    system=_CONTEXT_SYSTEM_PROMPT,
                    user=f"\n{"-"*80}\n\n".join(user_prompts),
                ),
                item=PaperOutput(
                    title=paper.title,
                    abstract=paper.abstract,
                    ratings=paper.ratings,
                    sections=paper.sections,
                    approval=paper.approval,
                    references=classified_references,
                ),
            )
        )

    assert len(paper_outputs) == len(papers)
    return GPTResult(paper_outputs, total_cost)


def _log_config(
    *,
    model: str,
    data_path: Path,
    limit_papers: int | None,
    limit_references: int | None,
    user_prompt: str,
    output_dir: Path,
    use_expanded_context: bool,
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
        f"  Use expanded context: {use_expanded_context}\n"
    )


def classify_contexts(
    model: str,
    api_key: str | None,
    data_path: Path,
    limit_papers: int | None,
    user_prompt_key: str,
    output_dir: Path,
    limit_references: int | None,
    use_expanded_context: bool,
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
        use_expanded_context=use_expanded_context,
    )

    client = OpenAI()

    data = TypeAdapter(list[PaperInput]).validate_json(data_path.read_bytes())

    papers = data[:limit_papers]
    user_prompt = _CONTEXT_USER_PROMPTS[user_prompt_key]

    with Timer() as timer:
        results = _classify_contexts(
            client, model, user_prompt, papers, limit_references, use_expanded_context
        )

    logger.info(f"Time elapsed: {timer.human}")
    logger.info(f"Total cost: ${results.cost:.10f}")

    contexts = [result.item for result in results.result]
    stats, metrics = _show_classified_stats(contexts)
    logger.info("Classification metrics:\n%s\n", stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_bytes(
        TypeAdapter(list[PromptResult[PaperOutput]]).dump_json(results.result, indent=2)
    )
    (output_dir / "output.txt").write_text(stats)
    if metrics is not None:
        (output_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))


# TODO:This one is a bit messy. Refactor it.
def _show_classified_stats(
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

    assert len(y_true) == len(y_pred)
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
            f" ({sum(y_true)/len(y_true):.2%})",
            f"Pred (P/N): {sum(y_pred)}/{len(y_pred) - sum(y_pred)}"
            f" ({sum(y_pred)/len(y_pred):.2%})",
        ]
        return "\n".join(output), metrics

    # No entries with gold annotation
    positive = sum(bool(context.prediction) for context in all_contexts)
    negative = len(all_contexts) - positive
    output += [
        "No gold values to calculate metrics.",
        f"Positive: {positive} ({positive / len(all_contexts):.2%})",
        f"Negative {negative} ({negative / len(all_contexts):.2%})",
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
        default="simple",
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
        "--use-expanded-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use expanded context (sentences surrounding the citation context)."
        " Defaults to %(default)s.",
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
    setup_logging("gpt")

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
            args.ref_limit,
            args.use_expanded_context,
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
