"""Construct and query SciMON graphs."""

import typer

from paper.baselines.scimon import build, citations, kg, query, semantic

app = typer.Typer(
    name="scimon",
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

subcommands = [
    ("kg", kg),
    ("semantic", semantic),
    ("citations", citations),
    ("build", build),
    ("query", query),
]
for name, module in subcommands:
    app.command(name=name, help=module.__doc__, no_args_is_help=True)(module.main)
