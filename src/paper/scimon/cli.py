"""CLI entrypoint for SciMON graph construction.

- kg: KG graph. Terms connected by directed used-for relations.
- semantic: semantic graph. Nodes connected by cosine similarity between embeddings from
  prompt-based based inputs.
- citations: citation graph. Papers are connected to their top K citations by cosine
  similarity between the titles.
- build: Build all three graphs and store them in a single object and file.

Embeddings are generated with SentenceTransformer. The default model is
`all-mpnet-base-v2`.
"""

import typer

from paper.scimon import build, citations, kg, semantic

app = typer.Typer(
    name="gpt",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)

subcommands = [
    ("kg", kg),
    ("semantic", semantic),
    ("citations", citations),
    ("build", build),
]
for name, module in subcommands:
    app.command(name=name, help=module.__doc__, no_args_is_help=True)(module.main)
