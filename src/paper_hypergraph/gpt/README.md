# GPT: Prompt based graph extraction

The goal is to give the LLM a paper's title, abstract and introduction, and extract the
following hierarchy:
- Root node: the title of the paper
- Intermediate nodes: the main concepts covered in the paper, from the abstract
- Leaves: sentences from the introduction mentioning the main concepts

This forms a hierarchical tree, as nodes only have edges to their children: the title node
to the concepts, and the concepts to their sentences.

*NOTE*: This is a proof of concept. The prompt and pipeline need to be refined.

## Usage

This script requires the `OPENAI_API_KEY` environment variable to be set, either through
the environment, the `.env` file at the root of the repository, or the `--api-key`
argument.

Run from the repository root:

```bash
$ uv run src/paper_hypergraph/gpt/extract_graph.py /path/to/asap_extracted.json
```

The default logging level is INFO, but there's some DEBUG output that can be useful.
Change the level using the `LOG_LEVEL` environment variable.

```bash
$ LOG_LEVEL=DEBUG uv run src/paper_hypergraph/gpt/extract_graph.py /path/to/asap_extracted.json
```
