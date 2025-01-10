# PeerRead processing

Merge the files from the papers in the PeerRead dataset into a single file. Note that
only entries that have ratings in their reviews will be used.

## Usage

- Download the dataset from Google Drive.
```console
$ uv run peerread download data/PeerRead
```
- Preprocess the dataset.
```console
$ uv run peerread preprocess data/PeerRead output/peerread_merged.json
```

The paths can be changed via CLI arguments. See `peerread --help` for more information.
