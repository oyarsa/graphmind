"""CLI entrypoint for GPT models and tasks.

- context: Run context classification on ASAP references.
- graph: Extract concept graph from an ASAP and perform classification on it.
- eval_full: Paper evaluation based on the full text only.
- tokens: Estimate input tokens from different prompts and demonstrations.
"""

import asyncio
import logging

from paper.gpt import (
    annotate_terms,
    classify_contexts,
    evaluate_paper_full,
    extract_graph,
    tokens,
)
from paper.util import HelpOnErrorArgumentParser, doc_summary, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)

    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    subcommands = [
        ("graph", "Extract graph from papers.", extract_graph),
        ("context", "Classify paper citations using full text.", classify_contexts),
        ("eval_full", "Evaluate paper using full text.", evaluate_paper_full),
        (
            "terms",
            "Annotate S2 papers with key terms for problems and methods",
            annotate_terms,
        ),
    ]
    for name, help, module in subcommands:
        cmd_parser = subparsers.add_parser(
            name, help=help, description=doc_summary(module)
        )
        module.setup_cli_parser(cmd_parser)

    tokens_parser = subparsers.add_parser(
        "tokens",
        help="Estimate input tokens for tasks and prompts.",
        description=tokens.__doc__,
    )
    tokens.setup_cli_parser(tokens_parser)

    args = parser.parse_args()
    setup_logging()

    if args.command == "graph":
        if args.subcommand == "prompts":
            extract_graph.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                extract_graph.extract_graph(
                    args.model,
                    args.data_path,
                    args.limit,
                    args.graph_user_prompt,
                    args.classify_user_prompt,
                    args.display,
                    args.output_dir,
                    args.classify,
                    args.continue_papers,
                    args.clean_run,
                    args.seed,
                )
            )

    elif args.command == "context":
        if args.subcommand == "prompts":
            classify_contexts.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                classify_contexts.classify_contexts(
                    args.model,
                    args.data_path,
                    args.limit,
                    args.user_prompt,
                    args.output_dir,
                    args.ref_limit,
                    args.continue_papers,
                    args.clean_run,
                    args.seed,
                )
            )

    elif args.command == "eval_full":
        if args.subcommand == "prompts":
            evaluate_paper_full.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                evaluate_paper_full.evaluate_papers(
                    args.model,
                    args.data_path,
                    args.limit,
                    args.user_prompt,
                    args.output_dir,
                    args.continue_papers,
                    args.clean_run,
                    args.seed,
                    args.demos,
                    args.demo_prompt,
                )
            )

    elif args.command == "tokens":
        if args.subcommand == "eval_full":
            tokens.fulltext(
                args.input_file,
                args.user_prompt_key,
                args.demo_prompt_key,
                args.demonstrations_file,
                args.model,
                args.limit,
            )
        else:  # other commands
            pass

    elif args.command == "terms":
        if args.subcommand == "prompts":
            annotate_terms.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                annotate_terms.annotate_papers_terms(
                    args.input_file,
                    args.output_dir,
                    args.limit,
                    args.model,
                    args.seed,
                    args.user_prompt,
                    args.continue_papers,
                    args.clean_run,
                )
            )


if __name__ == "__main__":
    main()
