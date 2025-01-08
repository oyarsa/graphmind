# `scimon`

Construct SciMON graphs.

**Usage**:

```console
$ scimon [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `build`: Build the three SciMON graphs (KG,...
* `citations`: Create citations graph with the reference...
* `kg`: Build a KG graph from extracted terms from...
* `query`: Query all terms for PeerRead-annotated...
* `semantic`: Build a semantic graph from extracted...

## `scimon build`

Build the three SciMON graphs (KG, semantic and citations) as a single structure.

The stored graph needs to be converted to a real one in memory because of how the
embeddings are stored.

This takes two inputs:
- Annotated papers wrapped in prompts (`gpt.PromptResult[gpt.PaperAnnotated]`) from
  `paper.gpt.annotate_paper`.
- PeerRead papers with full S2 reference data (`s2.PaperWithFullS2`) from
  `semantic_scholar.info`.

**Usage**:

```console
$ scimon build [OPTIONS]
```

**Options**:

* `--ann PATH`: File with annotated papers.  [required]
* `--peerread PATH`: File with PeerRead and references.  [required]
* `--output PATH`: Output file with the constructed graphs.  [required]
* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--test / --no-test`: Test graph saving and loading.  [default: no-test]
* `--help`: Show this message and exit.

## `scimon citations`

Create citations graph with the reference papers sorted by title similarity.

Calculates title sentence embedding for the PeerRead main paper and each S2 reference
using a SentenceTransformer, then keeping a sorted list by similarity. At query time,
the user can specify the number of most similar papers to retrieve. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `semantic_scholar.construct_daset`: the file
`peerread_with_s2_references.json` of type `peerread.PaperWithS2Refs`. The similarity is
calculated between the PeerRead `title` and the S2 `title_peer`.

**Usage**:

```console
$ scimon citations [OPTIONS] INPUT_FILE OUTPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: File with PeerRead papers with references with full S2 data.  [required]
* `OUTPUT_FILE`: File with PeerRead papers with top K references with full S2 data.  [required]

**Options**:

* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--help`: Show this message and exit.

## `scimon kg`

Build a KG graph from extracted terms from papers.

The input is the output of `paper.gpt.annotate_paper`. Since we use the direct output,
of the script, it's wrapped: `run_gpt.PromptResult[annotate_paper.PaperAnnotated]`.

Since we're interested on building the graphs from the relations, we ignore the terms
from `PaperAnnotated.GPTTerms`, and focus only on the relations. The terms are used
for the semantic graph (`paper.scimon.semantic`).

**Usage**:

```console
$ scimon kg [OPTIONS] INPUT_FILE OUTPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: Input file with the extracted terms.  [required]
* `OUTPUT_FILE`: Output file with the constructed KG graph.  [required]

**Options**:

* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--query TEXT`: Test query for the graph
* `--help`: Show this message and exit.

## `scimon query`

Query all terms for PeerRead-annotated papers and save the result as JSON.

The inputs are:
- Annotated PeerRead papers from `gpt.annotate_paper`.
- The SciMON graph created from annotated S2 papers (also `gpt.annotate_paper`) via
  `scimon.build`.

**Usage**:

```console
$ scimon query [OPTIONS]
```

**Options**:

* `--ann-peer PATH`: JSON file containing the annotated PeerRead papers data.  [required]
* `--graph PATH`: JSON file containing the SciMON graphs.  [required]
* `--output PATH`: Path to the output file with annotated papers and their graph results.  [required]
* `--help`: Show this message and exit.

## `scimon semantic`

Build a semantic graph from extracted terms and backgrounds from papers.

The input is the output of `paper.gpt.annotate_paper`. Since we use the direct output,
of the script, it's wrapped: `run_gpt.PromptResult[annotate_paper.PaperAnnotated]`.

**Usage**:

```console
$ scimon semantic [OPTIONS] INPUT_FILE OUTPUT_FILE
```

**Arguments**:

* `INPUT_FILE`: Input file with the extracted terms.  [required]
* `OUTPUT_FILE`: Output file with the constructed semantic graph.  [required]

**Options**:

* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--limit INTEGER`: Maximum number of papers to process.
* `--batch_size INTEGER`: Encoding batch size.  [default: 128]
* `--help`: Show this message and exit.
