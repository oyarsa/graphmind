"""Estimates the number of input tokens for a given dataset.

Supports the following modes:
- abstract: title and abstract-based paper evaluation
- scimon: SciMON graph-based paper evaluation

Todo:
- extract_graph: extract entity graph from paper text
- evaluate_paper_graph: graph-based paper evaluation

WON'T DO:
- classify_context: classify paper and context sentence into positive/negative. I won't
  do this because we already have the best configuration (short instructions with only
  the citation sentence) and it uses very few tokens.
"""

import logging
from pathlib import Path
from typing import Annotated

import tiktoken
import typer

from paper import semantic_scholar as s2
from paper.baselines import scimon
from paper.gpt.evaluate_paper import (
    Demonstration,
    format_demonstrations,
)
from paper.gpt.evaluate_paper_scimon import format_template as format_scimon
from paper.gpt.prompts import PromptTemplate
from paper.gpt.prompts.eval_demonstrations import EVALUATE_DEMONSTRATION_PROMPTS
from paper.gpt.prompts.evaluate_graph import GRAPH_EVAL_USER_PROMPTS
from paper.gpt.prompts.evaluate_paper_scimon import SCIMON_CLASSIFY_USER_PROMPTS
from paper.gpt.run_gpt import MODEL_SYNONYMS, MODELS_ALLOWED
from paper.util import cli, describe, display_params, setup_logging
from paper.util.serde import load_data

_GRAPH_VARIANTS = (
    "full-graph-structured",
    "full-text-only",
    "full-text",
)

logger = logging.getLogger(__name__)


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


def _format_abstract_template(
    prompt: PromptTemplate, paper: s2.PaperWithS2Refs, demonstrations: str
) -> str:
    """Format abstract-only prompt template with paper data."""
    return prompt.template.format(
        title=paper.title,
        abstract=paper.abstract or "",
        demonstrations=demonstrations,
    )


@app.command(
    name="abstract",
    help="Estimate tokens for title+abstract evaluation",
    no_args_is_help=True,
)
def abstract(
    input_file: Annotated[
        Path, typer.Argument(help="Input dataset JSON file with S2 paper data")
    ],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt (e.g., 'sans').",
            click_type=cli.Choice(GRAPH_EVAL_USER_PROMPTS),
        ),
    ] = "sans",
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo-prompt",
            help="Demonstration prompt.",
            click_type=cli.Choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ] = "abstract",
    demonstrations_file: Annotated[
        Path | None, typer.Option("--demo-file", help="Path to demonstrations file")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Which model's tokeniser to use.")
    ] = "gpt-4o-mini",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="Limit on the number of entities to process."
        ),
    ] = None,
) -> None:
    """Estimate tokens for title and abstract-based paper evaluation."""
    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    input_data = load_data(input_file, s2.PaperWithS2Refs)[:limit]
    input_prompt = GRAPH_EVAL_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
    )
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    prompts = [
        _format_abstract_template(input_prompt, paper, demonstrations)
        for paper in input_data
    ]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info("Token stats:\n%s\n", describe(tokens))


@app.command(
    name="scimon",
    help="Estimate tokens for SciMON-based evaluation",
    no_args_is_help=True,
)
def scimon_(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input dataset JSON file (annotated PeerRead with graph data.)"
        ),
    ],
    user_prompt_key: Annotated[
        str,
        typer.Option(
            "--user",
            help="Input data prompt.",
            click_type=cli.Choice(SCIMON_CLASSIFY_USER_PROMPTS),
        ),
    ],
    demo_prompt_key: Annotated[
        str,
        typer.Option(
            "--demo-prompt",
            help="Demonstration prompt.",
            click_type=cli.Choice(EVALUATE_DEMONSTRATION_PROMPTS),
        ),
    ],
    demonstrations_file: Annotated[
        Path | None, typer.Option("--demo-file", help="Path to demonstrations file")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Which model's tokeniser to use.")
    ] = "gpt-4o-mini",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", "-n", help="Limit on the number of entities to process."
        ),
    ] = None,
) -> None:
    """Estimate tokens for full text-based paper evaluation."""
    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    anns = load_data(input_file, scimon.AnnotatedGraphResult)[:limit]
    input_prompt = SCIMON_CLASSIFY_USER_PROMPTS[user_prompt_key]

    demonstration_data = (
        load_data(demonstrations_file, Demonstration) if demonstrations_file else []
    )
    demonstration_prompt = EVALUATE_DEMONSTRATION_PROMPTS[demo_prompt_key]
    demonstrations = format_demonstrations(demonstration_data, demonstration_prompt)

    prompts = [
        format_scimon(input_prompt, ann_result, demonstrations) for ann_result in anns
    ]

    tokeniser = tiktoken.encoding_for_model(model)
    tokens = [len(tokeniser.encode(prompt)) for prompt in prompts]
    logger.info("Token stats:\n%s\n", describe(tokens))


@app.command(
    name="graph",
    help="Estimate tokens for graph-based evaluation variants",
    no_args_is_help=True,
)
def graph(
    input_file: Annotated[
        Path, typer.Argument(help="Input dataset JSON file (PaperWithRelatedSummary)")
    ],
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Graph cache directory."),
    ] = Path("output/.cache/graphs"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Which model's tokeniser to use.")
    ] = "gpt-4o-mini",
    seed: Annotated[
        int, typer.Option("--seed", help="Seed used for cache lookup.")
    ] = 42,
    cache_key: Annotated[
        str | None,
        typer.Option(
            "--cache-key", help="Explicit graph cache key (subdirectory name)."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit on the number of papers to process."),
    ] = None,
) -> None:
    """Estimate tokens for graph-based evaluation variants on a dataset.

    Computes token statistics for:
    - full-graph-structured (GraphMind best variant)
    - full-text-only (full paper text, no related papers)
    - full-text with use_abstracts=False (full text + related summaries)
    - full-text with use_abstracts=True (full text + related raw abstracts)
    """
    from paper.gpt.evaluate_paper_graph import format_eval_template
    from paper.gpt.graph_cache import compute_cache_key, load_cached_graphs
    from paper.gpt.model import (
        Graph,
        LinearisationMethod,
        PaperWithRelatedSummary,
        PromptResult,
    )

    logger.info(display_params())

    model = MODEL_SYNONYMS.get(model, model)
    if model not in MODELS_ALLOWED:
        raise SystemExit(f"Invalid model: {model!r}. Must be one of: {MODELS_ALLOWED}.")

    wrapped = load_data(input_file, PromptResult[PaperWithRelatedSummary])[:limit]
    papers = PromptResult.unwrap(wrapped)
    tokeniser = tiktoken.encoding_for_model(model)

    # Load cached graphs
    if cache_key is None:
        cache_key = compute_cache_key(
            model=model,
            temperature=0.0,
            graph_prompt_key="excerpts",
            seed=seed,
            input_file=input_file,
        )
    cache_path = cache_dir / cache_key
    cached = load_cached_graphs(cache_path)
    if cached is None:
        raise SystemExit(
            f"No cached graphs at {cache_path}. Run graph extraction first."
        )
    logger.info("Loaded %d cached graphs from %s", len(cached), cache_path)

    # Map papers to their graphs
    empty_graph = Graph(title="", abstract="", entities=[], relationships=[])
    paper_graphs: list[tuple[PaperWithRelatedSummary, Graph]] = []
    missing = 0
    for paper in papers:
        graph_id = Graph(
            title=paper.title, abstract=paper.abstract, entities=[], relationships=[]
        ).id
        g = cached.get(graph_id)
        if g is None:
            missing += 1
            g = empty_graph
        paper_graphs.append((paper, g))

    if missing:
        logger.warning("%d papers had no cached graph (using empty graph)", missing)

    # Define variants: (label, prompt_key, use_abstracts, max_main_text_tokens)
    variants: list[tuple[str, str, bool, int | None]] = [
        ("GraphMind (full-graph-structured)", "full-graph-structured", False, None),
        ("Full text only", "full-text-only", False, None),
        ("Full text + summaries", "full-text", False, None),
        ("Full text + raw abstracts", "full-text", True, None),
        ("Full text only (truncated)", "full-text-only", False, 4800),
        ("Full text + summaries (truncated)", "full-text", False, 4800),
        ("Full text + raw abstracts (truncated)", "full-text", True, 4800),
    ]

    for label, prompt_key, use_abstracts, max_main_text_tokens in variants:
        prompt = GRAPH_EVAL_USER_PROMPTS[prompt_key]
        tokens: list[int] = []
        for paper, g in paper_graphs:
            text = format_eval_template(
                prompt,
                paper,
                g,
                demonstrations="",
                method=LinearisationMethod.TOPO,
                use_abstracts=use_abstracts,
                max_main_text_tokens=max_main_text_tokens,
            )
            system_tokens = len(tokeniser.encode(prompt.system or ""))
            user_tokens = len(tokeniser.encode(text))
            total = system_tokens + user_tokens

            tokens.append(total)

        logger.info("=== %s ===\n%s\n", label, describe(tokens))


@app.callback()
def main() -> None:
    """Set up logging."""
    setup_logging()
