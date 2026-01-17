"""Re-implemented baselines for comparison."""

import typer

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


try:
    from paper.baselines.scimon import cli as scimon

    app.add_typer(scimon.app, name="scimon")
except ImportError:
    pass


try:
    from paper.baselines import novascore

    app.add_typer(novascore.app, name="nova")
except ImportError:
    pass

try:
    from paper.baselines import sft

    app.add_typer(sft.app, name="sft")
except ImportError:
    pass

try:
    from paper.baselines import sft_gen

    app.add_typer(sft_gen.app, name="sft-gen")
except ImportError:
    pass

if __name__ == "__main__":
    app()
