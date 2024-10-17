# GPT: Using prompting to extract paper concepts and connections

## Graph extraction

The goal is to give the LLM a paper's title, abstract and introduction, and extract the
following hierarchy:
- Root node: the title of the paper
- Intermediate nodes: the main concepts covered in the paper, from the abstract
- Leaves: sentences from the introduction mentioning the main concepts

This forms a hierarchical DAG, as nodes only have edges to their children: the title node
to the concepts, and the concepts to their sentences.

*NOTE*: This is a proof of concept. The prompt and pipeline need to be refined.

### Usage

This script requires the `OPENAI_API_KEY` environment variable to be set, either through
the environment, the `.env` file at the root of the repository, or the `--api-key`
argument.

```console
$ uv run gpt graph run output/asap_filtered.json output/graph
```

The default logging level is INFO, but there's some DEBUG output that can be useful.
Change the level using the `LOG_LEVEL` environment variable.

```console
$ LOG_LEVEL=DEBUG uv run gpt graph run output/asap_filtered.json output/graph
```

## Context classification

One of the key elements of the graph is classifying the relation between the main paper
and its citations. Each citation has one or more accompanying contexts, which we can use
to determine the citation polarity: whether it was positive (the cited paper supports
the argument) or negative (the paper serves as contrast). We can also determine the
citation type: whether it concerns the method, results, contributions, etc.

For now, we explore this as a separate script, `classify_contexts.py`. Eventually, this
would be added to the main `extract_graph.py` script to be performed along the other
parts of the pipeline.


### Usage

This script also requires an OpenAI key. See above.

```console
$ uv run gpt context run output/asap_filtered.json output/context
```

## More information

See the following help commands for more information on using the tools:

```console
# General help
$ uv run gpt --help

# Graph generation tool
$ uv run gpt graph --help
$ uv run gpt graph run --help
$ uv run gpt graph prompts --help

# Context classification tool
$ uv run gpt context --help
$ uv run gpt context run --help
$ uv run gpt context prompts --help
```
