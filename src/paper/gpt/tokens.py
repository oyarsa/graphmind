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

import logging
from pathlib import Path
from typing import Annotated

import tiktoken
import typer

from paper import semantic_scholar as s2
from paper.baselines import scimon
from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS as DEMO_PROMPTS,
)
from paper.gpt.evaluate_paper import (
    Demonstration,
    format_demonstrations,
)
from paper.gpt.evaluate_paper_sans import SANS_CLASSIFY_USER_PROMPTS as SANS_PROMPTS
from paper.gpt.evaluate_paper_sans import format_template as format_sans
from paper.gpt.evaluate_paper_scimon import (
    SCIMON_CLASSIFY_USER_PROMPTS as SCIMON_PROMPTS,
)
from paper.gpt.evaluate_paper_scimon import format_template as format_scimon
from paper.gpt.run_gpt import MODEL_SYNONYMS, MODELS_ALLOWED
from paper.util import cli, describe, display_params, setup_logging
from paper.util.serde import load_data

logger = logging.getLogger(__name__)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(help="Estimate tokens for paper content evaluation", no_args_is_help=True)
def sans(
    input_file: Annotated[
        Path, typer.Argument(help="Input dataset JSON file (peerread_merged.json)")
    ],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt.",
            click_type=cli.Choice(SANS_PROMPTS),
        ),
    ],
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo-prompt",
            help="Demonstration prompt.",
            click_type=cli.Choice(DEMO_PROMPTS),
        ),
    ],
    demonstrations_file: Annotated[
        Path | None, typer.Option("--demo-file", help="Path to demonstrations file")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Which model's tokeniser to use.")
    ] = "gpt-4o-mini",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="Limit on the number of entities to process."
        ),
    ] = None,
) -> None:
    """Estimate tokens for full text-based paper evaluation."""
    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    input_data = load_data(input_file, s2.PaperWithS2Refs)[:limit]
    input_prompt = SANS_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
    )
    demonstration_prompt = DEMO_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    prompts = [format_sans(input_prompt, paper, demonstrations) for paper in input_data]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info("Token stats:\n%s\n", describe(tokens))


@app.command(
    name="scimon",
    help="Estimate tokens for SciMON-based evaluation",
    no_args_is_help=True,
)
def scimon_(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input dataset JSON file (annotated PeerRead with graph data.)"
        ),
    ],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt.",
            click_type=cli.Choice(SCIMON_PROMPTS),
        ),
    ],
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo-prompt",
            help="Demonstration prompt.",
            click_type=cli.Choice(DEMO_PROMPTS),
        ),
    ],
    demonstrations_file: Annotated[
        Path | None, typer.Option("--demo-file", help="Path to demonstrations file")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Which model's tokeniser to use.")
    ] = "gpt-4o-mini",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="Limit on the number of entities to process."
        ),
    ] = None,
) -> None:
    """Estimate tokens for full text-based paper evaluation."""
    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    anns = load_data(input_file, scimon.AnnotatedGraphResult)[:limit]
    input_prompt = SCIMON_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
    )
    demonstration_prompt = DEMO_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    prompts = [
        format_scimon(input_prompt, ann_result, demonstrations) for ann_result in anns
    ]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info("Token stats:\n%s\n", describe(tokens))


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()
