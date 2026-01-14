"""Tool commands for the paper project."""

import typer

from paper.tools import exp_log

app = typer.Typer(
    help="Miscellaneous tools",
    no_args_is_help=True,
)

app.add_typer(exp_log.app, name="explog")

