"""Entrypoint for ASAP and S2ORC preprocessing pipelines."""

import argparse

from paper.asap import preprocess as asap
from paper.s2orc import preprocess as s2orc


def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Run different preprocessing pipelines"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Choose the pipeline to run"
    )

    # Add S2ORC subcommand using its existing cli_parser as a parent
    s2orc_parser = s2orc.cli_parser()
    subparsers.add_parser(
        "s2orc",
        parents=[s2orc_parser],
        help="Run S2ORC preprocessing pipeline",
        add_help=False,
    )

    # Add ASAP-Review subcommand using its existing cli_parser as a parent
    asap_parser = asap.cli_parser()
    subparsers.add_parser(
        "asap",
        parents=[asap_parser],
        help="Run ASAP-Review preprocessing pipeline",
        add_help=False,
    )

    args = parser.parse_args()

    if args.command == "s2orc":
        s2orc.pipeline(
            args.output_path, args.api_key, args.output_path, args.file_limit
        )
    elif args.command == "asap":
        asap.pipeline(args.input, args.output, args.max_papers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
