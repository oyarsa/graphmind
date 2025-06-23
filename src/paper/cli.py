"""Command-line interface for the paper project providing access to all tools."""

import typer

from paper import construct_dataset, demo_data, find_type, split
from paper.baselines import cli as baselines
from paper.deps import pipeline_viz
from paper.gpt import cli as gpt
from paper.orc import cli as orc
from paper.peerread import cli as peerread
from paper.peter import cli as peter
from paper.semantic_scholar import cli as s2

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
app.add_typer(split.app, name="split")

app.command(no_args_is_help=True, name="construct")(construct_dataset.main)
app.command(no_args_is_help=True, name="findtype")(find_type.main)
app.command(no_args_is_help=True, name="demo_data")(demo_data.main)
app.command(no_args_is_help=False, name="deps")(pipeline_viz.main)


if __name__ == "__main__":
    app()
