"""Tool commands for the paper project."""

import typer

from paper.tools import demo_data, exp_log, find_type, pipeline_viz, split

app = typer.Typer(
    help="Miscellaneous tools",
    no_args_is_help=True,
)

app.add_typer(exp_log.app, name="explog")
app.add_typer(split.app, name="split")

app.command(no_args_is_help=True, name="findtype")(find_type.main)
app.command(no_args_is_help=True, name="demodata")(demo_data.main)
app.command(no_args_is_help=False, name="deps")(pipeline_viz.main)
