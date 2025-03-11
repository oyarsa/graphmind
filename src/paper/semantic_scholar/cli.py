"""Prompt-based tools for paper evaluation, annotation and tasks using the OpenAI API."""

import logging

import typer

from paper.semantic_scholar import areas, info, recommended

logger = logging.getLogger(__name__)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Fetch information about papers from the Semantic Scholar API.",
)
app.add_typer(info.app, name="info")
app.command(no_args_is_help=True, name="areas")(areas.main)
app.command(no_args_is_help=True, name="recommended")(recommended.main)
