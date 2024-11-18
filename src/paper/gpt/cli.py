"""CLI entrypoint for GPT models and tasks.

- context: Run context classification on ASAP references.
- graph: Extract concept graph from an ASAP and perform classification on it.
- eval_full: Paper evaluation based on the full text only.
- tokens: Estimate input tokens from different prompts and demonstrations.
"""

import logging

import typer

from paper.gpt import (
    annotate_paper,
    classify_contexts,
    evaluate_paper_full,
    extract_graph,
    tokens,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

subcommands = [
    ("graph", "Extract graph from papers.", extract_graph),
    ("context", "Classify paper citations using full text.", classify_contexts),
    ("eval_full", "Evaluate paper using full text.", evaluate_paper_full),
    (
        "terms",
        "Annotate S2 papers with key terms and split abstract.",
        annotate_paper,
    ),
    ("tokens", "Estimate input tokens for tasks and prompts.", tokens),
]
for name, help, module in subcommands:
    app.add_typer(
        module.app,
        name=name,
        short_help=help,
        help=module.__doc__.splitlines()[0] if module.__doc__ else None,
    )
