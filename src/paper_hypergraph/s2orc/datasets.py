import requests


def main() -> None:
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


if __name__ == "__main__":
    main()
