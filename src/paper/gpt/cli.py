"""Prompt-based tools for paper evaluation, annotation and tasks using the OpenAI API."""

import logging
from collections.abc import Iterable
from types import ModuleType

import typer

from paper.gpt import (
    annotate_paper,
    classify_contexts,
    evaluate_paper,
    evaluate_paper_graph,
    evaluate_paper_peter,
    evaluate_paper_sans,
    evaluate_paper_scimon,
    evaluate_paper_search,
    evaluate_rationale,
    evaluate_reviews,
    extract_acu,
    summarise_related_peter,
    tokens,
)

logger = logging.getLogger(__name__)


def _new_app(
    app_name: str, subcommands: Iterable[tuple[str, str, ModuleType]]
) -> typer.Typer:
    """Create new Typer app with commands from the `subcommands` modules.

    Each entry in `subcommands` should have:
    - name of the module
    - help text
    - the module itself, which should have an `app` object and an optional docstring
    """
    app = typer.Typer(
        name=app_name,
        context_settings={"help_option_names": ["-h", "--help"]},
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
        no_args_is_help=True,
        help=__doc__,
    )
    for name, help, module in subcommands:
        app.add_typer(
            module.app,
            name=name,
            short_help=help,
            help=module.__doc__.splitlines()[0] if module.__doc__ else None,
        )
    return app


main_subcommands = [
    ("context", "Classify paper citations using full text.", classify_contexts),
    (
        "terms",
        "Annotate S2 papers with key terms and split abstract.",
        annotate_paper,
    ),
    ("tokens", "Estimate input tokens for tasks and prompts.", tokens),
    ("petersum", "Summarise PETER related papers.", summarise_related_peter),
    ("acus", "Extract Atomic Content Units from related papers", extract_acu),
]
app = _new_app("gpt", main_subcommands)

evals_subcommands = [
    ("sans", "Evaluate paper using just the paper contents.", evaluate_paper_sans),
    (
        "scimon",
        "Evaluate paper using SciMON graphs-extracted terms.",
        evaluate_paper_scimon,
    ),
    (
        "peter",
        "Evaluate paper using PETER-query related papers.",
        evaluate_paper_peter,
    ),
    (
        "graph",
        "Evaluate paper using paper graph with PETER-query related papers.",
        evaluate_paper_graph,
    ),
    (
        "search",
        "Evaluate paper using LLM grounded with web search.",
        evaluate_paper_search,
    ),
    ("reviews", "Evaluate individual reviews for novelty.", evaluate_reviews),
    (
        "rationale",
        "Evaluate generated rationales from graph evaluation.",
        evaluate_rationale,
    ),
]

app.add_typer(
    _new_app("eval", evals_subcommands),
    name="eval",
    help="Evaluate papers novelty ratings.",
)


@app.command(help="List available demonstration files.")
def demos() -> None:
    """Print the available demonstration file names."""
    for name in evaluate_paper.EVALUATE_DEMONSTRATIONS:
        print(f"- {name}")
