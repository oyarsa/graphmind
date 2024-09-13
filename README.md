# Paper Hypergraph: evaluating paper novelty and feasibility

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/). When running commads, use `uv run`:

```bash
uv run python example.py [..args]
```

No need to set up a virtual environment or install dependencies. `uv run` will take care
of that automatically.

## Components

- `./src/paper_hypergraph/s2orc`: download, extract and process data from the Semantic
  Scholar Open Research Corpus (S2ORC) dataset. See the README and `preprocess.py` for
  instructions on how to download and extract the data.
- `./src/paper_hypergraph/asap`: filter and process papers from the ASAP-Review dataset.
  See the README.md for instructions to download the data.
