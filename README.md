# Paper Hypergraph: evaluating paper novelty and feasibility

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/). When running commads, use `uv run`:

```bash
uv run python example.py [..args]
```

No need to set up a virtual environment or install dependencies. `uv run` will take care
of that automatically.

## Components

### Datasets

- `./src/paper_hypergraph/s2orc`: download, extract and process data from the Semantic
  Scholar Open Research Corpus (S2ORC) dataset. See the README and `preprocess.py` for
  instructions on how to download and extract the data.
- `./src/paper_hypergraph/asap`: filter and process papers from the ASAP-Review dataset.
  See the README.md for instructions to download the data.

In each case, see the README and `preprocess.py` for instructions on how to download and
extract the data.

You can also use the `preprocess` command to access to both the S2ORC and ASAP dataset
preprocessing:

```bash
# Preprocess S2ORC dataset
uv run preprocess s2orc
# Preprocess ASAP dataset
uv run preprocess asap
# More information
uv run preprocess --help
```
