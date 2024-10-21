# Paper Hypergraph: evaluating paper novelty and feasibility

## Getting started

> [!NOTE]
> This repository has been tested on macOS and Ubuntu. It should work on other Linux
> systems, but it has not been tested on Windows. It should work on WSL.

To automatically set up the environment, run:

```bash
./tasks.sh setup
```

See [Development Environment](#development-environment) and
[CONTRIBUTING](/CONTRIBUTING.md) for more information.

After this setup, you're reading to process the datasets. See [Datasets](#datasets)
below.

## Datasets

- **S2ORC - Semantic Scholar Open Research Corpus**: dataset with the full paper content
  from Semantic Scholar. See [s2orc README](/src/paper_hypergraph/s2orc/README.md) for
  information.
- **ASAP-Review**: dataset with full content and ratings for ICLR 2017-2022 papers. See
  [asap README](/src/paper_hypergraph/asap/README.md) for more information.

Use the `uv run preprocess` command to access both S2ORC and ASAP dataset preprocessing:

```console
# Create `data` and `output` directories
$ mkdir data output

# > Preprocess ASAP dataset
# Download the dataset from Google Drive: https://drive.usercontent.google.com/download?id=1nJdljy468roUcKLbVwWUhMs7teirah75&export=download&authuser=0
# Extract to `data/asap`.
# Output will be saved to `output`. The final file is `output/asap_filtered.json`.
$ uv run preprocess asap data/asap output

# > Preprocess S2ORC dataset
# Note: this takes a long time, potentially hours.
# This needs a Semantic Scholar API, which you can get from https://www.semanticscholar.org/product/api#api-key-form
# Downloaded S2ORC data and intermediate files will be stored on `data/s2orc`
# Output will be saved to `output/s2orc_papers.json.gz`
$ uv run preprocess s2orc data/s2orc output --api-key YOUR_SEMANTIC_SCHOLAR_KEY
# You can also set the SEMANTIC_SCHOLAR_API_KEY environment variable instead of using
# `--api-key`.

# More information on the commands and options
$ uv run preprocess s2orc --help
$ uv run preprocess asap --help
$ uv run preprocess --help
```

Both S2ORC and ASAP have multi-step pre-processing pipelines. The commands above will
run all of them in sequence from scratch. To run individual commands (e.g. during
testing), see the respective READMEs.

## Graph Generation

Once you have set up the environment and processed the datasets, you can run the graph
generation tool:

```console
# Generate the graphs from ASAP papers
$ uv run graph-gpt run output/asap_filtered output/graph --api-key YOUR_OPENAI_KEY
# You can also set the OPENAI_API_KEY environment variable instead of using `--api-key`

# See the available prompts
$ uv run graph-gpt prompts

# For more information on the available options
$ uv run graph-gpt --help
$ uv run graph-gpt run --help
$ uv run graph-gpt prompts --help
```

## Development Environment

When running commands, use `uv run`:

```bash
uv run python example.py [...args]
```

You don't have to set up a virtual environment or install dependencies. `uv run` will
take care of that automatically.

If you're running Python scripts or commands (e.g. see `preprocess` below), you can omit
the `python`. Real example:

```bash
uv run src/paper_hypergraph/s2orc/acl.py
```

See `./tasks.sh` for common development tasks. In particular, use `./tasks.sh check` to
run the linter, formatter and type checker. Run `./tasks.sh help` to see all tasks.

The `./tasks.sh setup` task sets up the full environment:

- [`uv`](https://docs.astral.sh/uv/): manages the project, Python environment and
  dependencies, including the development tools.
- [`ruff`](https://docs.astral.sh/ruff/): linter and formatter.
- [`pyright`](https://microsoft.github.io/pyright): type-checker.
- [`pre-commit`](https://pre-commit.com/): which runs some basic checks before you
  create a commit.

Please check the individual tool documentation for more information. See also
[CONTRIBUTING](/CONTRIBUTING.md) for in-depth information.

To view the project documentation in the browser (module, functions, class, etc.), run

```bash
./tasks.sh doc
```

## License

This project is licensed under the GPL v3 or later:

    paper-hypergraph: evaluating paper novelty and feasibility
    Copyright (C) 2024 The paper-hypergraph contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

See the [LICENSE](LICENSE) file for the full license text. See the
[CONTRIBUTORS](CONTRIBUTORS) file for a list of contributors.
