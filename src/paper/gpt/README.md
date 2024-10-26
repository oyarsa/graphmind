# GPT: Using prompting to extract paper concepts and connections

## Graph extraction

Given the title, abstract and main text of a paper, we want to extract information to
represent the important aspects of the paper as a graph.

Connections:
- Paper title —> TLDR (1:1)
- Paper title —> Primary area (1:1)
- Paper title —> Keywords (1:N, N <= 5)
- TLDR —> Claims (1:N)
- Claims —> Methods (N:M)
- Methods —> Experiments (N:M)

Definitions:
- Primary area: pick from a [list of options from
  ICLR](https://iclr.cc/Conferences/2025/CallForPapers).
- Keywords: no more than five.
- TLDR: a sentence that summarises the paper.
- Claims: summarise what the paper claims to contribute, especially claims made in the
  abstract, introduction, discussion and conclusion. Pay attention to key phrases that
  highlight new findings or interpretations.
- Methods: for each claim, you identify the methods used to validate it from the method
  sections (pay attention to the difference between methods and experiments). These
  include the key components: algorithms, theoretical framework, modification or novel
  techniques introduced.
- Experiments: models, baselines, datasets, etc., used in experiments to validate the
  methods. We don’t need experiment results, just configuration/test environment.

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

Run `uv run gpt graph --help` to see more information about the commands.

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

This will save the following files in `output/context`:
- `output.txt`: text report on metrics, like the one printed at the end of the execution
- `result.json`: after exeuction completes, this is the full result saved
- `result.tmp.json`: results saved during execution in case something happens to
  interrupt it

If something happens, it's possible to continue the execution from where it last stopped
using the flag `--continue-papers path/to/result.tmp.json`. This will avoid re-processing
data.

Run `uv run gpt context --help` to see more information about the commands.

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
