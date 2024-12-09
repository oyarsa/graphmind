"""CLI entrypoint for GPT models and tasks.

- context: Run context classification on ASAP references.
- graph: Extract concept graph from an ASAP and perform classification on it.
- eval_full: Paper evaluation based on the full text only.
- tokens: Estimate input tokens from different prompts and demonstrations.
"""

import logging
from collections.abc import Iterable
from types import ModuleType

import typer

from paper.gpt import (
    annotate_paper,
    classify_contexts,
    evaluate_paper_full,
    evaluate_paper_peter,
    evaluate_paper_scimon,
    extract_graph,
    summarise_related_peter,
    tokens,
)

logger = logging.getLogger(__name__)


def _new_app(
    app_name: str, subcommands: Iterable[tuple[str, str, ModuleType]]
) -> typer.Typer:
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
    ("graph", "Extract graph from papers.", extract_graph),
    ("context", "Classify paper citations using full text.", classify_contexts),
    (
        "terms",
        "Annotate S2 papers with key terms and split abstract.",
        annotate_paper,
    ),
    ("tokens", "Estimate input tokens for tasks and prompts.", tokens),
    ("petersum", "Summarise PETER related papers.", summarise_related_peter),
]
app = _new_app("gpt", main_subcommands)

evals_subcommands = [
    ("full", "Evaluate paper using full text.", evaluate_paper_full),
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
]

app.add_typer(
    _new_app("eval", evals_subcommands),
    name="eval",
    help="Evaluate a paper",
)
