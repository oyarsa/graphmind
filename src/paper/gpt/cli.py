"""Prompt-based tools for paper evaluation, annotation and tasks using the OpenAI API."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from types import ModuleType

import typer

from paper.gpt import (
    annotate_paper,
    classify_contexts,
    evaluate_paper,
    evaluate_paper_graph,
    evaluate_paper_peter,
    evaluate_paper_sans,
    evaluate_paper_scimon,
    evaluate_paper_search,
    evaluate_rationale,
    evaluate_reviews,
    evaluate_tournament,
    extract_acu,
    extract_paper_graph,
    run_gpt,
    summarise_related_peter,
    tokens,
)
from paper.gpt.single_paper import single_paper

logger = logging.getLogger(__name__)


def _new_app(
    app_name: str, subcommands: Iterable[tuple[str, str, ModuleType]]
) -> typer.Typer:
    """Create new Typer app with commands from the `subcommands` modules.

    Each entry in `subcommands` should have:
    - name of the module
    - help text
    - the module itself, which should have an `app` object and an optional docstring
    """
    app = typer.Typer(
        name=app_name,
        context_settings={"help_option_names": ["-h", "--help"]},
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
        no_args_is_help=True,
        help=__doc__,
    )
    for name, help, module in subcommands:
        app.add_typer(
            module.app,
            name=name,
            short_help=help,
            help=module.__doc__.splitlines()[0] if module.__doc__ else None,
        )
    return app


main_subcommands = [
    ("context", "Classify paper citations using full text.", classify_contexts),
    (
        "terms",
        "Annotate S2 papers with key terms and split abstract.",
        annotate_paper,
    ),
    ("tokens", "Estimate input tokens for tasks and prompts.", tokens),
    ("petersum", "Summarise PETER related papers.", summarise_related_peter),
    ("acus", "Extract Atomic Content Units from related papers", extract_acu),
    ("graph", "Extract hierarchical graph from paper contents", extract_paper_graph),
]
app = _new_app("gpt", main_subcommands)

evals_subcommands = [
    ("sans", "Evaluate paper using just the paper contents.", evaluate_paper_sans),
    (
        "scimon",
        "Evaluate paper using SciMON graphs-extracted terms.",
        evaluate_paper_scimon,
    ),
    (
        "peter",
        "Evaluate paper using PETER-query related papers.",
        evaluate_paper_peter,
    ),
    (
        "graph",
        "Evaluate paper using paper graph with PETER-query related papers.",
        evaluate_paper_graph,
    ),
    (
        "search",
        "Evaluate paper using LLM grounded with web search.",
        evaluate_paper_search,
    ),
    ("reviews", "Evaluate individual reviews for novelty.", evaluate_reviews),
    (
        "rationale",
        "Evaluate generated rationales from graph evaluation.",
        evaluate_rationale,
    ),
    (
        "rationale-tournament",
        "Evaluate generated rationales from graph evaluation using pairwise tournament.",
        evaluate_tournament.cli,
    ),
]

app.add_typer(
    _new_app("eval", evals_subcommands),
    name="eval",
    help="Evaluate papers novelty ratings.",
)


@app.command(help="List available demonstration files.")
def demos() -> None:
    """Print the available demonstration file names."""
    for name in evaluate_paper.EVALUATE_DEMONSTRATIONS:
        print(f"- {name}")


@app.command(help="List available models.")
def models() -> None:
    """Print the available models."""
    reverse_mapping: dict[str, list[str]] = defaultdict(list)
    for shortcut, full_name in run_gpt.MODEL_SYNONYMS.items():
        reverse_mapping[full_name].append(shortcut)

    for model in sorted(set(run_gpt.MODEL_SYNONYMS.values())):
        shortcuts = reverse_mapping[model]
        shortcuts_str = ", ".join(s for s in shortcuts if s != model)

        print(f"- {model}")
        if shortcuts_str:
            print(f"  Alias: {shortcuts_str}")

        # Add pricing information if available
        if model in run_gpt.MODEL_COSTS:
            input_cost, output_cost = run_gpt.MODEL_COSTS[model]
            print(f"  Price (in/out): ${input_cost:.2f}/1M, ${output_cost:.2f}/1M")

        print()


app.command(name="single", no_args_is_help=True)(single_paper)

if __name__ == "__main__":
    app()
