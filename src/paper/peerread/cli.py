"""PeerRead dataset commands."""

import typer

from paper.peerread.download import download
from paper.peerread.preprocess import preprocess

app = typer.Typer(
    name="peerread",
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help="Run PeerRead dataset commands.",
)

app.command(name="download", help="Download PeerRead dataset.", no_args_is_help=True)(
    download
)
app.command(
    name="preprocess", help="Run PeerRead preprocessing pipeline.", no_args_is_help=True
)(preprocess)


if __name__ == "__main__":
    app()
