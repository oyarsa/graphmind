"""Estimates the number of input tokens for a given dataset.

Supports the following modes:
- evaluate_paper_full: full text-based paper evaluation

Todo:
- extract_graph: extract entity graph from paper text
- evaluate_paper_graph: graph-based paper evaluation

WON'T DO:
- classify_context: classify paper and context sentence into positive/negative. I won't
  do this because we already have the best configuration (short instructions with only
  the citation sentence) and it uses very few tokens.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore
import tiktoken
from pydantic import TypeAdapter

from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS as DEMO_PROMPTS,
)
from paper.gpt.evaluate_paper import (
    Demonstration,
    format_demonstrations,
)
from paper.gpt.evaluate_paper_full import FULL_CLASSIFY_USER_PROMPTS as FULLTEXT_PROMPTS
from paper.gpt.evaluate_paper_full import format_template as format_fulltext
from paper.gpt.model import Paper
from paper.gpt.run_gpt import MODEL_SYNONYMS, MODELS_ALLOWED
from paper.util import HelpOnErrorArgumentParser, display_params, setup_logging

logger = logging.getLogger(__name__)


def fulltext(
    input_file: Path,
    user_prompt_key: str,
    demo_prompt_key: str,
    demonstrations_file: Path | None,
    model: str,
    limit: int | None,
) -> None:
    """Estimate tokens for full text-based paper evaluation."""
    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    input_data = TypeAdapter(list[Paper]).validate_json(input_file.read_bytes())[:limit]
    input_prompt = FULLTEXT_PROMPTS[user_prompt_key]

    demonstration_data = (
        TypeAdapter(list[Demonstration]).validate_json(demonstrations_file.read_bytes())
        if demonstrations_file is not None
        else []
    )
    demonstration_prompt = DEMO_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    prompts = [
        format_fulltext(input_prompt, paper, demonstrations) for paper in input_data
    ]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info(
        "Token stats:\n%s\n",
        pd.Series(tokens).describe().astype(int).to_string(),  # type: ignore
    )


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    setup_cli_parser(parser)
    setup_logging()

    args = parser.parse_args()
    if args.subcommand == "eval_full":
        fulltext(
            args.input_file,
            args.user_prompt_key,
            args.demo_prompt_key,
            args.demonstrations_file,
            args.model,
            args.limit,
        )


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", required=True
    )

    # Fulltext subcommand
    parser_fulltext = subparsers.add_parser(
        "eval_full", help="Full text-based paper evaluation"
    )
    parser_fulltext.add_argument(
        "input_file", type=Path, help="Input dataset JSON file (asap_filtered.json)"
    )
    parser_fulltext.add_argument(
        "--user",
        dest="user_prompt_key",
        choices=sorted(FULLTEXT_PROMPTS),
        required=True,
        help="Input data prompt. Required.",
    )
    parser_fulltext.add_argument(
        "--demo",
        dest="demo_prompt_key",
        choices=sorted(DEMO_PROMPTS),
        required=True,
        help="Demonstration prompt. Required.",
    )
    parser_fulltext.add_argument(
        "--demos",
        dest="demonstrations_file",
        type=Path,
        help="Path to demonstrations file",
    )
    parser_fulltext.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help="Which model's tokeniser to use. Default %(default)s",
    )
    parser_fulltext.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Number of entries to process. Default: all",
    )


if __name__ == "__main__":
    main()
