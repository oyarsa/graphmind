# `gpt`

CLI entrypoint for GPT models and tasks.

- context: Run context classification on ASAP references.
- graph: Extract concept graph from an ASAP and perform classification on it.
- eval_full: Paper evaluation based on the full text only.
- tokens: Estimate input tokens from different prompts and demonstrations.

**Usage**:

```console
$ gpt [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `context`: Classify paper citations using full text.
* `eval_full`: Evaluate paper using full text.
* `graph`: Extract graph from papers.
* `terms`: Annotate S2 papers with key terms and split abstract.
* `tokens`: Estimate input tokens for tasks and prompts.

## `gpt context`

Classify paper citation contexts by polarity using GPT-4.

**Usage**:

```console
$ gpt context [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Classify paper citation contexts by...

### `gpt context prompts`

List available prompts.

**Usage**:

```console
$ gpt context prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

### `gpt context run`

Classify paper citation contexts by polarity using GPT-4.

Each paper contains many references. These can appear in one or more citation contexts
inside the text. Each citation context can be classified by polarity (positive vs
negative).

NB: We do concurrent requests (mediated by a rate limiter). Unfortunately, that doesn't
work very well with OpenAI's client. This means you'll likely see a lot of
openai.APIConnectionError thrown around. Most requests will go through, so you'll just
have to run the script again until you get everything. See also the `--continue-papers`
option.

**Usage**:

```console
$ gpt context run [OPTIONS] DATA_PATH OUTPUT_DIR
```

**Arguments**:

* `DATA_PATH`: The path to the JSON file containing the papers data.  [required]
* `OUTPUT_DIR`: The path to the output directory where files will be saved.  [required]

**Options**:

* `-m, --model [4o|4o-mini|gpt-4o|gpt-4o-2024-08-06|gpt-4o-mini|gpt-4o-mini-2024-07-18]`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 1]
* `--user-prompt [full|sentence|simple]`: The user prompt to use for context classification.  [default: sentence]
* `--ref-limit INTEGER`: Limit to the number of references per paper to process.
* `--continue-papers PATH`: Path to file with data from a previous run
* `--clean-run / --no-clean-run`: Start from scratch, ignoring existing intermediate results  [default: no-clean-run]
* `--seed INTEGER`: Seed to set in the OpenAI call.  [default: 0]
* `--help`: Show this message and exit.

## `gpt eval_full`

Evaluate a paper's approval based on its full-body text.

**Usage**:

```console
$ gpt eval_full [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Evaluate a paper's approval based on its...

### `gpt eval_full prompts`

List available prompts.

**Usage**:

```console
$ gpt eval_full prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

### `gpt eval_full run`

Evaluate a paper's approval based on its full-body text.

**Usage**:

```console
$ gpt eval_full run [OPTIONS] DATA_PATH OUTPUT_DIR
```

**Arguments**:

* `DATA_PATH`: The path to the JSON file containing the papers data.  [required]
* `OUTPUT_DIR`: The path to the output directory where the files will be saved.  [required]

**Options**:

* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 1]
* `--user-prompt [iclr2023|iclr2023-abs|simple|simple-abs]`: The user prompt to use for classification.  [default: simple]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--clean-run / --no-clean-run`: Start from scratch, ignoring existing intermediate results  [default: no-clean-run]
* `--seed INTEGER`: Random seed used for data shuffling.  [default: 0]
* `--demos PATH`: File containing demonstrations to use in few-shot prompt
* `--demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations.  [default: simple]
* `--help`: Show this message and exit.

## `gpt graph`

Generate graphs from papers from the ASAP-Review dataset using OpenAI GPT.

**Usage**:

```console
$ gpt graph [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Generate graphs from papers from the...

### `gpt graph prompts`

List available prompts.

**Usage**:

```console
$ gpt graph prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

### `gpt graph run`

Generate graphs from papers from the ASAP-Review dataset using OpenAI GPT.

The graphs represent the collection of concepts and arguments in the paper.
Can also classify a paper into approved/not-approved using the generated graph.

**Usage**:

```console
$ gpt graph run [OPTIONS] DATA_PATH OUTPUT_DIR
```

**Arguments**:

* `DATA_PATH`: The path to the JSON file containig the papers data.  [required]
* `OUTPUT_DIR`: The path to the output directory where files will be saved.  [required]

**Options**:

* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `--limit INTEGER`: Limit to the number of papers to process.
* `--graph-user-prompt [strict]`: The user prompt to use for the graph extraction.  [default: simple]
* `--classify-user-prompt [simple]`: The user prompt to use for paper classification.  [default: simple]
* `--display / --no-display`: Display the extracted graph  [default: no-display]
* `--classify / --no-classify`: Classify the papers based on the extracted entities.  [default: no-classify]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--clean-run / --no-clean-run`: Start from scratch, ignoring existing intermediate results.  [default: no-clean-run]
* `--seed INTEGER`: Seed to set in the OpenAI call.  [default: 0]
* `--help`: Show this message and exit.

## `gpt terms`

Extract key terms for problems and methods from S2 Papers.

**Usage**:

```console
$ gpt terms [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Extract key terms for problems and methods...

### `gpt terms prompts`

List available prompts.

**Usage**:

```console
$ gpt terms prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

### `gpt terms run`

Extract key terms for problems and methods from S2 Papers.

Input is a JSON array of `paper.external_data.semantic_scholar.model.Paper`. Output
contains the input paper plus the prompts used and the extracted terms.

**Usage**:

```console
$ gpt terms run [OPTIONS] INPUT_FILE OUTPUT_DIR
```

**Arguments**:

* `INPUT_FILE`: The path to the JSON file containing the papers data (S2Paper format).  [required]
* `OUTPUT_DIR`: The path to the output directory where files will be saved.  [required]

**Options**:

* `-n, --limit INTEGER`: The number of papers to process. Set to 0 for all papers.  [default: 0]
* `-m, --model [4o|4o-mini|gpt-4o|gpt-4o-2024-08-06|gpt-4o-mini|gpt-4o-mini-2024-07-18]`: The model to use for the annotation.  [default: gpt-4o-mini]
* `--seed INTEGER`: Seed to set in the OpenAI call.  [default: 0]
* `--prompt-term [multi]`: User prompt to use for term annotation.  [default: multi]
* `--prompt-abstract [simple]`: User prompt to use for abstract classification.  [default: simple]
* `--abstract-demos PATH`: Path to demonstrations containing data for abstract classification.
* `--abstract-demo-prompt [simple]`: Prompt used to create abstract classification demonstrations  [default: simple]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--clean-run / --no-clean-run`: Start from scratch, ignoring existing intermediate results.  [default: no-clean-run]
* `--log [none|table|detail]`: How much detail to show in output logging.  [default: none]
* `--help`: Show this message and exit.

## `gpt tokens`

Estimates the number of input tokens for a given dataset.

**Usage**:

```console
$ gpt tokens [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `fulltext`: Estimates the number of input tokens for a...

### `gpt tokens fulltext`

Estimates the number of input tokens for a given dataset.

Supports the following modes:
- evaluate_paper_full: full text-based paper evaluation

Todo:
- extract_graph: extract entity graph from paper text
- evaluate_paper_graph: graph-based paper evaluation

WON'T DO:
- classify_context: classify paper and context sentence into positive/negative. I won't
  do this because we already have the best configuration (short instructions with only
  the citation sentence) and it uses very few tokens.

**Usage**:

```console
$ gpt tokens fulltext [OPTIONS] INPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: Input dataset JSON file (asap_filtered.json)  [required]

**Options**:

* `--user [iclr2023|iclr2023-abs|simple|simple-abs]`: Input data prompt.  [required]
* `--demo [abstract|maintext]`: Demonstration prompt.  [required]
* `--demonstrations-file PATH`: Path to demonstrations file
* `-m, --model TEXT`: Which model's tokeniser to use.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: Limit on the number of entities to process.
* `--help`: Show this message and exit.
