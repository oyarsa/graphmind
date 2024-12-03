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

import pandas as pd  # type: ignore
import tiktoken
import typer

from paper.gpt.evaluate_paper import (
    EVALUATE_DEMONSTRATION_PROMPTS as DEMO_PROMPTS,
)
from paper.gpt.evaluate_paper import (
    Demonstration,
    format_demonstrations,
)
from paper.gpt.evaluate_paper_full import FULL_CLASSIFY_USER_PROMPTS as FULLTEXT_PROMPTS
from paper.gpt.evaluate_paper_full import format_template as format_fulltext
from paper.gpt.evaluate_paper_scimon import (
    SCIMON_CLASSIFY_USER_PROMPTS as SCIMON_PROMPTS,
)
from paper.gpt.evaluate_paper_scimon import format_template as format_scimon
from paper.gpt.evaluate_paper_scimon import query_papers
from paper.gpt.model import ASAPAnnotated, Paper
from paper.gpt.run_gpt import MODEL_SYNONYMS, MODELS_ALLOWED
from paper.scimon.graph import graph_from_json
from paper.util import cli, display_params, setup_logging
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


@app.command(help="Estimate tokens for full-text evaluation", no_args_is_help=True)
def fulltext(
    input_file: Annotated[
        Path, typer.Argument(help="Input dataset JSON file (asap_filtered.json)")
    ],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt.",
            click_type=cli.choice(FULLTEXT_PROMPTS),
        ),
    ],
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo",
            help="Demonstration prompt.",
            click_type=cli.choice(DEMO_PROMPTS),
        ),
    ],
    demonstrations_file: Annotated[
        Path | None, typer.Option(help="Path to demonstrations file")
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

    input_data = load_data(input_file, Paper)[:limit]
    input_prompt = FULLTEXT_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
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


@app.command(help="Estimate tokens for SciMON-based evaluation", no_args_is_help=True)
def scimon(
    input_file: Annotated[
        Path, typer.Argument(help="Input dataset JSON file (annotated ASAP)")
    ],
    graph_file: Annotated[Path, typer.Argument(help="Path to SciMON graph")],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt.",
            click_type=cli.choice(SCIMON_PROMPTS),
        ),
    ],
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo",
            help="Demonstration prompt.",
            click_type=cli.choice(DEMO_PROMPTS),
        ),
    ],
    demonstrations_file: Annotated[
        Path | None, typer.Option(help="Path to demonstrations file")
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

    anns = load_data(input_file, ASAPAnnotated)[:limit]
    input_prompt = SCIMON_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
    )
    demonstration_prompt = DEMO_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    graph = graph_from_json(graph_file)

    prompts = [
        format_scimon(input_prompt, ann_result, demonstrations)
        for ann_result in query_papers(graph, anns)
    ]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info(
        "Token stats:\n%s\n",
        pd.Series(tokens).describe().astype(int).to_string(),  # type: ignore
    )


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()
