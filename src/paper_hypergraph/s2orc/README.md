# S2ORC: scripts to manipulate Semantic Scholar's API

## Usage

This is the data pipeline:

1. [`download.py`](download.py): download S2ORC dataset. Downloads about 270
   files totalling 250 GB.
2. [`extract.py`](extract.py): extract data from downloaded S2ORC dataset.
   Extracts each file to memory, one at a time and extracts the relevant information
   from it. Note: uses about 4 GB of memory per file.
3. [`acl_papers.py`](acl_papers.py): get papers whose venue matches certain ACL
   conference keywords (e.g. "ACL", "Association for Computation Lingusitics", etc.)

For information on how to run each script, run `uv run <script>.py --help`.

These are in separate scripts because they are long-running tasks. `download.py`
has to download 250 GB of data, and `extract.py` has to extract 1 TB of data to
memory (one file at a time) and extract the relevant information from it.

You can also run `preprocess.py` to run all the scripts in order, but I recommend running
them separately.

The scripts need an API key. You can obtain one from the [Semantic Scholar
website](https://www.semanticscholar.org/product/api#api-key-form). The scripts require
the key to either be in the `SEMANTIC_SCHOLAR_API_KEY` environment variable or passed
as a CLI argument (see the `--help` for each script).

### Dealing with JSON.GZ files

All scripts read gzipped files to memory and save .json.gz because of storage
limitations. This doesn't seem to impact write/read times significantly.

- In Python, you can use the `gzip` module and the `gzip.open` from the standard library
  to open and read/write to the file as if it were a normal file, including using
  `json.load` and `json.dump`. Note that you have to use `rt` or `wt` as the mode. Refer
  to `extract.py` for an example.
- In the command line, you can combine `gzip` and `jq` to manipulate the files. Example:
  `gzip -dc file.json.gz | jq map(.venue)`.

## Unused

The following scripts are not used in the pipeline:

- [`datasets.py`](datasets.py): list datasets in Semantic Scholar's API.
- [`filesizes.py`](filesizes.py): list filesizes of S2ORC dataset.
- [`acl.py`](acl.py): script to download ACL papers from Semantic Scholar's API.
