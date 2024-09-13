# S2ORC: scripts to manipulate Semantic Scholar's API

## Usage

This is the data pipeline:

1. [`download_s2orc.py`](download_s2orc.py): download S2ORC dataset. Downloads about 270
   files totalling 250 GB.
2. [`process_s2orc.py`](process_s2orc.py): extract data from downloaded S2ORC dataset.
   Extracts each file to memory, one at a time and extracts the relevant information
   from it. Note: uses about 4 GB of memory per file.
3. [`unique_venues.py`](unique_venues.py): extract unique venues from S2ORC dataset.
4. [`match_venues.py`](match_venues.py): match venues from a list of ACL keywords and
   writes the names of the matching venues to a file. The list of venues comes from
   `unique_venues.py`.
5. [`acl_papers.py`](acl_papers.py): get papers matching a list of venues. The list of
   venues comes from `match_venues.py`.

These are in separate scripts because they are long-running tasks. `download_s2orc.py`
has to download 250 GB of data, and `process_s2orc.py` has to extract 1 TB of data to
memory (one file at a time) and extract the relevant information from it.

All scripts read gzipped files to memory and save .json.gz because of storage
limitations. This doesn't seem to impact write/read times significantly.

### Dealing with JSON.GZ files

- In Python, you can use the `gzip` module and the `gzip.open` from the standard library
  to open and read/write to the file as if it were a normal file, including using
  `json.load` and `json.dump`. Note that you have to use `rt` or `wt` as the mode. Refer
  to `process_s2orc.py` for an example.
- In the command line, you can combine `gzip` and `jq` to manipulate the files. Example:
  `gzip -dc file.json.gz | jq map(.venue)`.

## Unused

The following scripts are not used in the pipeline:

- [`datasets.py`](datasets.py): list datasets in Semantic Scholar's API.
- [`filesizes.py`](filesizes.py): list filesizes of S2ORC dataset.
- [`acl.py`](acl.py): script to download ACL papers from Semantic Scholar's API.
