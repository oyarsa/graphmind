"""Command-line interface for the paper project providing access to all tools."""

import typer

from paper import orc
from paper.baselines.scimon import cli as scimon
from paper.gpt import cli as gpt
from paper.peerread import cli as peerread
from paper.peter import cli as peter

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Paper analysis and processing tools.",
)
app.add_typer(peerread.app, name="peerread")
app.add_typer(gpt.app, name="gpt")
app.add_typer(scimon.app, name="scimon")
app.add_typer(peter.app, name="peter")
app.add_typer(orc.app, name="orc")


if __name__ == "__main__":
    app()
