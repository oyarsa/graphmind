"""List datasets and descriptions for the Semantic Scholar datasets API."""

import requests


def list_datasets() -> None:
    """List all datasets available from the Semantic Scholar datasets API."""
    releases_response = requests.get(
        "https://api.semanticscholar.org/datasets/v1/release/latest"
    )
    releases_response.raise_for_status()
    releases = releases_response.json()

    release_id = releases["release_id"]
    print(f"Latest release ID: {release_id}")
    endpoint = f"https://api.semanticscholar.org/datasets/v1/release/{release_id}"

    datasets_response = requests.get(endpoint)
    datasets_response.raise_for_status()
    data = datasets_response.json()

    for dataset in data["datasets"]:
        print(dataset["name"], "-", dataset["description"])
        print()


def main() -> None:
    list_datasets()


if __name__ == "__main__":
    main()
