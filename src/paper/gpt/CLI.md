# `gpt`

CLI entrypoint for GPT models and tasks.

- context: Run context classification on PeerRead references.
- graph: Extract concept graph from an PeerRead and perform classification on it.
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
* `demos`: List available demonstration files.
* `eval`: Evaluate a paper
* `petersum`: Summarise PETER related papers.
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

Here, these references contain data from the S2 API, so we want to keep that, in addition
to the context and its class.

Data:
- input: s2.PaperWithS2Refs
- output: PaperWithContextClassfied

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
* `--continue`: Use existing intermediate results
* `--seed INTEGER`: Seed to set in the OpenAI call.  [default: 0]
* `--help`: Show this message and exit.

## `gpt demos`

List available demonstration files.

**Usage**:

```console
$ gpt demos [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `gpt eval`

Evaluate a paper

**Usage**:

```console
$ gpt eval [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `graph`: Evaluate paper using paper graph with PETER-query related papers.
* `peter`: Evaluate paper using PETER-query related papers.
* `reviews`: Evaluate individual reviews for novelty.
* `sans`: Evaluate paper using just the paper contents.
* `scimon`: Evaluate paper using SciMON graphs-extracted terms.

### `gpt eval graph`

Evaluate a paper's novelty based on main paper graph with PETER-queried papers.

**Usage**:

```console
$ gpt eval graph [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Evaluate a paper's novelty based on main...

#### `gpt eval graph prompts`

List available prompts.

**Usage**:

```console
$ gpt eval graph prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full prompt text.  [default: no-detail]
* `--help`: Show this message and exit.

#### `gpt eval graph run`

Evaluate a paper's novelty based on main paper graph with PETER-queried papers.

The input is the output of `gpt.summarise_related_peter`. These are the PETER-queried
papers with the related papers summarised. This then converts the paper content to a
graph, and uses it as input alongside the PETER results.

The output is the input annotated papers with a predicted novelty rating.

**Usage**:

```console
$ gpt eval graph run [OPTIONS]
```

**Options**:

* `--papers PATH`: JSON file containing the annotated PeerRead papers with summarised graph results.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for both extraction and evaluation.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 10]
* `--eval-prompt [full-graph|only-graph|related|sans|title-graph]`: The user prompt to use for paper evaluation.  [default: simple]
* `--graph-prompt [strict]`: The user prompt to use for graph extraction.  [default: strict]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
* `--seed INTEGER`: Random seed used for data shuffling and OpenAI API.  [default: 0]
* `--demos [eval_10|eval_4|review_10|review_5|review_clean_5|review_clean_5_1]`: Name of file containing demonstrations to use in few-shot prompt.
* `--demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations.  [default: abstract]
* `--help`: Show this message and exit.

### `gpt eval peter`

Evaluate a paper's novelty based on annotated papers with PETER-queried papers.

**Usage**:

```console
$ gpt eval peter [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Evaluate a paper's novelty based on...

#### `gpt eval peter prompts`

List available prompts.

**Usage**:

```console
$ gpt eval peter prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

#### `gpt eval peter run`

Evaluate a paper's novelty based on annotated papers with PETER-queried papers.

The input is the output of `gpt.summarise_related_peter`. This are the PETER-queried
papers with the related papers summarised.

The output is the input annotated papers with a predicted novelty rating.

**Usage**:

```console
$ gpt eval peter run [OPTIONS]
```

**Options**:

* `--papers PATH`: JSON file containing the annotated PeerRead papers with summarised graph results.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 10]
* `--user-prompt [sans|simple]`: The user prompt to use for classification.  [default: simple]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
* `--seed INTEGER`: Random seed used for data shuffling.  [default: 0]
* `--demos [eval_10|eval_4|review_10|review_5|review_clean_5|review_clean_5_1]`: Name of file containing demonstrations to use in few-shot prompt
* `--demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations.  [default: abstract]
* `--help`: Show this message and exit.

### `gpt eval reviews`

Predict each review's novelty rating based on the review text.

**Usage**:

```console
$ gpt eval reviews [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Predict each review's novelty rating based...

#### `gpt eval reviews prompts`

List available prompts.

**Usage**:

```console
$ gpt eval reviews prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

#### `gpt eval reviews run`

Predict each review's novelty rating based on the review text.

The input is the processed PeerRead dataset (peerread.Paper).

**Usage**:

```console
$ gpt eval reviews run [OPTIONS]
```

**Options**:

* `--peerread PATH`: The path to the JSON file containing the PeerRead papers data.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process. Use 0 for all papers.  [default: 1]
* `--user-prompt [simple|ternary]`: The user prompt to use for classification.  [default: simple]
* `--extract-prompt [basic|overall|simple]`: The user prompt to use for novelty rationale extraction. If not provided, use the original rationale.
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
* `--seed INTEGER`: Random seed used for the GPT API and to shuffle the data.  [default: 0]
* `--demos [eval_10|eval_4|review_10|review_5|review_clean_5|review_clean_5_1]`: Name of file containing demonstrations to use in few-shot prompt.
* `--review-demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations for review classification.  [default: abstract]
* `--extract-demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations for rationale extraction.
* `--mode [original|binary|trinary]`: Which mode to apply to target ratings. See `paper.evaluate_paper.RatingMode`.  [default: binary]
* `--keep-intermediate / --no-keep-intermediate`: Keep intermediate results.  [default: no-keep-intermediate]
* `--help`: Show this message and exit.

### `gpt eval sans`

Evaluate a paper's novelty based only on its content.

**Usage**:

```console
$ gpt eval sans [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Evaluate a paper's novelty based only on...

#### `gpt eval sans prompts`

List available prompts.

**Usage**:

```console
$ gpt eval sans prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

#### `gpt eval sans run`

Evaluate a paper's novelty based only on its content.

Uses the paper title, abstract and, optionally, the main text.
"sans" means "without", meaning "without other features".

The input is the processed PeerRead dataset (peerread.Paper).

**Usage**:

```console
$ gpt eval sans run [OPTIONS]
```

**Options**:

* `--peerread PATH`: The path to the JSON file containing the PeerRead papers data.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 1]
* `--user-prompt [iclr2023|iclr2023-abs|simple|simple-abs]`: The user prompt to use for classification.  [default: simple-abs]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results
* `--seed INTEGER`: Random seed used for data shuffling.  [default: 0]
* `--demos [eval_10|eval_4|review_10|review_5|review_clean_5|review_clean_5_1]`: Name of file containing demonstrations to use in few-shot prompt
* `--demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations.  [default: abstract]
* `--help`: Show this message and exit.

### `gpt eval scimon`

Evaluate a paper's novelty based on annotated papers with SciMON-derived terms.

**Usage**:

```console
$ gpt eval scimon [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Evaluate a paper's novelty based on...

#### `gpt eval scimon prompts`

List available prompts.

**Usage**:

```console
$ gpt eval scimon prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

#### `gpt eval scimon run`

Evaluate a paper's novelty based on annotated papers with SciMON-derived terms.

The input is the output of `scimon.query`, i.e. the output of `gpt.annotate_paper`
(papers with extracted scientific terms) with the related terms extracted through the
SciMON graph created by `scimon.build`.

**Usage**:

```console
$ gpt eval scimon run [OPTIONS]
```

**Options**:

* `--ann-graph PATH`: JSON file containing the annotated PeerRead papers with graph results.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of papers to process.  [default: 1]
* `--user-prompt [simple]`: The user prompt to use for classification.  [default: simple]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
* `--seed INTEGER`: Random seed used for data shuffling.  [default: 0]
* `--demos PATH`: File containing demonstrations to use in few-shot prompt
* `--demo-prompt [abstract|maintext]`: User prompt to use for building the few-shot demonstrations.  [default: abstract]
* `--help`: Show this message and exit.

## `gpt petersum`

Summarise related papers from PETER for inclusion in main paper evaluation prompt.

**Usage**:

```console
$ gpt petersum [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `prompts`: List available prompts.
* `run`: Run related paper summarisation.

### `gpt petersum prompts`

List available prompts.

**Usage**:

```console
$ gpt petersum prompts [OPTIONS]
```

**Options**:

* `--detail / --no-detail`: Show full description of the prompts.  [default: no-detail]
* `--help`: Show this message and exit.

### `gpt petersum run`

Summarise related papers from PETER for inclusion in main paper evaluation prompt.

The input is the output of `paper.peerread`, i.e. the output of `gpt.annotate_paper`
(papers with extracted scientific terms) and `gpt.classify_contexts` (citations
contexts classified by polarity) with the related papers queried through the PETER
graph.

The output is similar to the input, but the related papers have extra summarised
information that can be useful for evaluating papers.

**Usage**:

```console
$ gpt petersum run [OPTIONS]
```

**Options**:

* `--ann-graph PATH`: JSON file containing the annotated PeerRead papers with graph results.  [required]
* `--output PATH`: The path to the output directory where the files will be saved.  [required]
* `-m, --model TEXT`: The model to use for the extraction.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: The number of PeerRead papers to process.  [default: 10]
* `--positive-prompt [negative|positive]`: The summarisation prompt to use for positively related papers.  [default: positive]
* `--negative-prompt [negative|positive]`: The summarisation prompt to use for negatively related papers.  [default: negative]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
* `--seed INTEGER`: Random seed used for data shuffling.  [default: 0]
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

Input is a JSON array of `paper.semantic_scholar.model.Paper`. Output contains the input
paper plus the prompts used and the extracted terms.

**Usage**:

```console
$ gpt terms run [OPTIONS] INPUT_FILE OUTPUT_DIR
```

**Arguments**:

* `INPUT_FILE`: The path to the JSON file containing the papers data (S2Paper format).  [required]
* `OUTPUT_DIR`: The path to the output directory where files will be saved.  [required]

**Options**:

* `--paper-type [s2|peerread]`: Type of paper for the input data  [required]
* `-n, --limit INTEGER`: The number of papers to process. Set to 0 for all papers.  [default: 0]
* `-m, --model [4o|4o-mini|gpt-4o|gpt-4o-2024-08-06|gpt-4o-mini|gpt-4o-mini-2024-07-18]`: The model to use for the annotation.  [default: gpt-4o-mini]
* `--seed INTEGER`: Seed to set in the OpenAI call.  [default: 0]
* `--prompt-term [multi]`: User prompt to use for term annotation.  [default: multi]
* `--prompt-abstract [simple]`: User prompt to use for abstract classification.  [default: simple]
* `--abstract-demos PATH`: Path to demonstrations containing data for abstract classification.
* `--abstract-demo-prompt [simple]`: Prompt used to create abstract classification demonstrations  [default: simple]
* `--continue-papers PATH`: Path to file with data from a previous run.
* `--continue`: Use existing intermediate results.
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

* `sans`: Estimate tokens for paper content evaluation
* `scimon`: Estimate tokens for SciMON-based evaluation

### `gpt tokens sans`

Estimate tokens for paper content evaluation

**Usage**:

```console
$ gpt tokens sans [OPTIONS] INPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: Input dataset JSON file (peerread_merged.json)  [required]

**Options**:

* `--user [iclr2023|iclr2023-abs|simple|simple-abs]`: Input data prompt.  [required]
* `--demo-prompt [abstract|maintext]`: Demonstration prompt.  [required]
* `--demo-file PATH`: Path to demonstrations file
* `-m, --model TEXT`: Which model's tokeniser to use.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: Limit on the number of entities to process.
* `--help`: Show this message and exit.

### `gpt tokens scimon`

Estimate tokens for SciMON-based evaluation

**Usage**:

```console
$ gpt tokens scimon [OPTIONS] INPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: Input dataset JSON file (annotated PeerRead with graph data.)  [required]

**Options**:

* `--user [simple]`: Input data prompt.  [required]
* `--demo-prompt [abstract|maintext]`: Demonstration prompt.  [required]
* `--demo-file PATH`: Path to demonstrations file
* `-m, --model TEXT`: Which model's tokeniser to use.  [default: gpt-4o-mini]
* `-n, --limit INTEGER`: Limit on the number of entities to process.
* `--help`: Show this message and exit.
