# GraphMind: evaluating paper novelty and feasibility

## Getting started

> [!NOTE]
> This repository has been tested on macOS and Ubuntu. It should work on other Linux
> systems, but it has not been tested on Windows. It should work on WSL.

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the development
environment and dependencies:

```console
# Set up virtual environment and install dependencies
$ uv sync --extra cpu --extra baselines  # For CPU-only PyTorch/SentenceTransformers/etc.
$ uv sync --extra cuda --extra baselines # For CUDA (x86-64 Linux only).
# Install pre-commit hooks
$ uv run pre-commit install
```

> [!IMPORTANT]
> For development, you must install the `baselines` extra to ensure all dependencies are
> available. Without it, `just lint` and other development commands will fail.

See [Development Environment](#development-environment) and
[CONTRIBUTING](/CONTRIBUTING.md) for more information.

[`pandoc`](https://pandoc.org/installing.html) is required to parse the LaTeX files for
the ORC dataset.

After this setup, you're reading to process the datasets. See [Datasets](#datasets)
below.

Docker and Docker Compose are required for the REST API. See [REST
API](./src/paper/backend/README.md) for more information.

## Datasets

- **PeerRead**: dataset with full content and ratings for ACL and ICLR papers. See the
  [PeerRead README](/src/paper/peerread/README.md) for more information.
- **CSAbstruct**: abstract classification dataset we use for demonstrations in
  [gpt.annotate_paper](/src/paper/gpt/demonstrations.py). See
  [their repository](https://github.com/allenai/sequential_sentence_classification/tree/cf5ad6c663550dd8203f148cd703768d9ee86ff4)

Use the `uv run peerread` command for PeerRead dataset preprocessing:

```console
# > Preprocess PeerRead dataset
# Download the dataset from GitHub.
$ uv run paper peerread download data/PeerRead
# Output will be saved to `output`. The final file is `output/peerread_merged.json`.
$ uv run paper peerread preprocess data/peerread output

# More information on the commands and options
$ uv run paper peerread --help
```

## Graph Generation

Once you have set up the environment and processed the datasets, you can run the graph
generation tool:

```console
# Generate the graphs from PeerRead papers
$ export OPENAI_API_KEY=...   # set the environment variable or use the .env file
$ export OPENAI_BASE_URL=...  # optional, if using an alternative API
$ uv run paper gpt graph run output/peerread_merged output/graph

# See the available prompts
$ uv run paper graph prompts

# For more information on the available options
$ uv run paper gpt graph --help
$ uv run paper gpt graph run --help
$ uv run paper gpt graph prompts --help
```

For more information, see the documentation for each module:

- [`peerread/README.md`](./src/paper/peerread/README.md).
- [`baselines/scimon/README.md`](./src/paper/baselines/scimon/README.md).
- [`peter/README.md`](./src/paper/peter/README.md).
- [`gpt/README.md`](./src/paper/gpt/README.md).

## Development Environment

When running commands, use `uv run`:

```bash
uv run python example.py [...args]
```

You don't have to set up a virtual environment or install dependencies. `uv run` will
take care of that automatically.

If you're running Python scripts or commands (e.g. see `peerread` above), you can omit
the `python`. Example:

```console
$ uv run src/paper/construct_dataset.py  # also `paper construct`
$ uv run paper peerread
```

See [uv's documentation](https://docs.astral.sh/uv/concepts/projects/run/) for more
information.

See `just` for common development tasks. In particular, use `just lint` to
run the linter, formatter and type checker. Run `just` to see all tasks.

We use the following tools for development:

- [`uv`](https://docs.astral.sh/uv/): manages the project, Python environment and
  dependencies, including the development tools.
- [`ruff`](https://docs.astral.sh/ruff/): linter and formatter.
- [`pyright`](https://microsoft.github.io/pyright): type-checker.
- [`pre-commit`](https://pre-commit.com/): runs some basic checks before you
  create a commit.

Please check the individual tool documentation for more information. See also
[CONTRIBUTING](/CONTRIBUTING.md) for in-depth information.

The project has rich documentation on every public item. You can use `pydoc` to get the
documentation for an item:

```console
$ uv run -m pydoc paper.peter.Graph
```

## REST API

We offer a REST API for paper evaluation. While the main program is focused on batched
paper annotation and evaluation, the REST API allows the user to search a paper on arXiv
and use the full process to evaluate it.

See [REST API](./src/paper/backend/README.md) for more information.

## License

This project is licensed under the AGPL v3 or later:

    graphmind: evaluating paper novelty and feasibility
    Copyright (C) 2024-2025 Italo Luis da Silva

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

See the [LICENSE](LICENSE) file for the full license text.
