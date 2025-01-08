"""Entrypoint for PeerRead and S2ORC preprocessing pipelines."""

import typer

from paper.peerread import preprocess as peerrread
from paper.s2orc import preprocess as s2orc

app = typer.Typer(
    name="preprocess",
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Run different preprocessing pipelines.",
)

app.command(
    name="s2orc", help="Run S2ORC preprocessing pipeline", no_args_is_help=True
)(s2orc.pipeline)
app.command(
    name="peerread", help="Run PeerRead preprocessing pipeline", no_args_is_help=True
)(peerrread.pipeline)


if __name__ == "__main__":
    app()
