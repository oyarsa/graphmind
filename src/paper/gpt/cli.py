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
from paper.util import HelpOnErrorArgumentParser, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)

    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

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

    # 'evaluate_paper_full'
    eval_full_parser = subparsers.add_parser(
        "eval_full",
        help="Run paper evaluation from full text",
        description="Run paper evaluation from full text with the provided arguments",
    )
    evaluate_paper_full.setup_cli_parser(eval_full_parser)

    # 'tokens'
    tokens_parser = subparsers.add_parser(
        "tokens",
        help="Estimate input tokens for task and prompts",
        description="Estimate input tokens for task and prompts with provided arguments",
    )
    tokens.setup_cli_parser(tokens_parser)

    # 'annotate_terms'
    annotate_parser = subparsers.add_parser(
        "annotate_terms",
        help="Annotate S2 Papers with key terms for problems and methods",
        description="Annotate S2 Papers with key terms with provided arguments",
    )
    annotate_terms.setup_cli_parser(annotate_parser)

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


if __name__ == "__main__":
    main()
