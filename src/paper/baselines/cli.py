"""Re-implemented baselines for comparison."""

import typer

from paper.baselines import novascore, sft
from paper.baselines.scimon import cli as scimon

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

app.add_typer(scimon.app, name="scimon")
app.add_typer(novascore.app, name="nova")
app.add_typer(sft.app, name="sft")

if __name__ == "__main__":
    app()
