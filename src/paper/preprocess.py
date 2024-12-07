"""Entrypoint for ASAP and S2ORC preprocessing pipelines."""

import typer

from paper.asap import preprocess as asap
from paper.s2orc import preprocess as s2orc

app = typer.Typer(
    name="preprocess",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Run different preprocessing pipelines.",
)

app.command(
    name="s2orc", help="Run S2ORC preprocessing pipeline", no_args_is_help=True
)(s2orc.pipeline)
app.command(
    name="asap", help="Run ASAP-Review preprocessing pipeline", no_args_is_help=True
)(asap.pipeline)


if __name__ == "__main__":
    app()
