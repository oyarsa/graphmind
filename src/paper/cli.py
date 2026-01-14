"""Command-line interface for the paper project providing access to all tools."""

import typer

from paper import construct_dataset, single_paper
from paper.baselines import cli as baselines
from paper.gpt import cli as gpt
from paper.orc import cli as orc
from paper.peerread import cli as peerread
from paper.peter import cli as peter
from paper.semantic_scholar import cli as s2
from paper.tools import cli as tools
from paper.util import VERSION

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Paper analysis and processing tools.",
)
app.add_typer(peerread.app, name="peerread")
app.add_typer(gpt.app, name="gpt")
app.add_typer(baselines.app, name="baselines")
app.add_typer(peter.app, name="peter")
app.add_typer(orc.app, name="orc")
app.add_typer(s2.app, name="s2")
app.add_typer(tools.app, name="tools")

app.command(no_args_is_help=True, name="construct")(construct_dataset.main)
app.command(no_args_is_help=False, name="single")(single_paper.main)


def _version_callback(value: bool) -> None:
    if value:
        print(f"paper {VERSION}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the application version and exit.",
    ),
) -> None:
    """My awesome CLI application."""


if __name__ == "__main__":
    app()
