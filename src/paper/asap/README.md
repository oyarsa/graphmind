# ASAP-Review processing

Merge the files from the papers in the ASAP-Review dataset into a single file. Note that
only entries that have ratings in their reviews will be used.

## Usage

- Download the dataset from Google Drive.
```bash
$ uv run src/paper/asap/download.py data/asap
```
- Run the merge script on the ICLR dataset. Saves a single `output.json` file with the
merged data.
```bash
$ uv run merge.py data/asap output/asap_merged.json
```
- Run the extraction script to get the relevant information from the merged data.
```bash
$ uv run extract.py output/asap_merged.json output/asap_extracted.json
```

The paths can be changed via CLI arguments. See `merge.py --help` and `extract.py
--help` for more information on the options.

You can also run the scripts in a single command:
```bash
$ uv run preprocess.py data/asap output/asap_extracted.json
```

In the ASAP-Review dataset, NIPS papers don't have ratings in their reviews, only the
ICLR papers.

See [`dataset.md`](dataset.md) for the original README file for the dataset.
