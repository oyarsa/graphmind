"""CLI entrypoint for GPT models and tasks.

- context: Run context classification on ASAP references.
- graph: Extract concept graph from an ASAP and perform classification on it.
"""

import argparse
import asyncio
import logging

from paper.gpt import classify_contexts, extract_graph
from paper.util import setup_logging

logger = logging.getLogger("paper.gpt.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for 'graph' and 'context' subcomands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Valid commands",
        dest="command",
        required=True,
    )

    # 'graph' subcommand parser
    graph_parser = subparsers.add_parser(
        "graph",
        help="Run graph extraction",
        description="Run graph extraction with the provided arguments.",
    )
    extract_graph.setup_cli_parser(graph_parser)

    # 'context' subcommand parser
    context_parser = subparsers.add_parser(
        "context",
        help="Run context classification",
        description="Run context classification with the provided arguments.",
    )
    classify_contexts.setup_cli_parser(context_parser)

    args = parser.parse_args()

    setup_logging()

    if args.command == "graph":
        if args.subcommand == "prompts":
            extract_graph.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                extract_graph.extract_graph(
                    args.model,
                    args.api_key,
                    args.data_path,
                    args.limit,
                    args.graph_user_prompt,
                    args.classify_user_prompt,
                    args.display,
                    args.output_dir,
                    args.classify,
                    args.continue_papers,
                )
            )

    elif args.command == "context":
        if args.subcommand == "prompts":
            classify_contexts.list_prompts(detail=args.detail)
        elif args.subcommand == "run":
            asyncio.run(
                classify_contexts.classify_contexts(
                    args.model,
                    args.api_key,
                    args.data_path,
                    args.limit,
                    args.user_prompt,
                    args.output_dir,
                    args.ref_limit,
                    args.continue_papers,
                )
            )


if __name__ == "__main__":
    main()
