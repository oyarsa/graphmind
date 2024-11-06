"""List datasets and descriptions for the Semantic Scholar datasets API."""

import argparse
import asyncio
import json

import aiohttp

from paper.util import HelpOnErrorArgumentParser


async def list_datasets(show_json: bool) -> None:
    async with aiohttp.ClientSession() as session:
        # Get latest release ID
        async with session.get(
            "https://api.semanticscholar.org/datasets/v1/release/latest"
        ) as releases_response:
            releases_response.raise_for_status()
            releases = await releases_response.json()

        release_id = releases["release_id"]
        print(f"Latest release ID: {release_id}")
        endpoint = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}"

        # Get datasets for the latest release
        async with session.get(endpoint) as datasets_response:
            datasets_response.raise_for_status()
            data = await datasets_response.json()

    if show_json:
        print(json.dumps(data["datasets"], indent=2))
    else:
        for dataset in data["datasets"]:
            print(dataset["name"], "-", dataset["description"])
            print()


async def main() -> None:
    parser = HelpOnErrorArgumentParser(__doc__)
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        help="Output data in JSON format instead of plain text",
    )
    args = parser.parse_args()
    await list_datasets(args.json)


if __name__ == "__main__":
    asyncio.run(main())
