# `peter`

Create PETER graphs.

**Usage**:

```console
$ peter [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `asap`: Query the graph with ASAP papers and save...
* `build`: Create full graph.
* `citations`: Create citations graph.
* `query`: Demonstrate the graph by querying it to...
* `semantic`: Create semantic graph.

## `peter asap`

Query the graph with ASAP papers and save the results in a file.

The output file contains both the original ASAP paper and the graph query results.

**Usage**:

```console
$ peter asap [OPTIONS]
```

**Options**:

* `--graph PATH`: Path to full graph file.  [required]
* `--asap-ann PATH`: File with ASAP papers with extracted backgrounds and targets.  [required]
* `--output PATH`: Output file to save the result data.  [required]
* `-n, --num-papers INTEGER`: Number of papers to query. Defaults to all papers.
* `--help`: Show this message and exit.

## `peter build`

Create full graph.

**Usage**:

```console
$ peter build [OPTIONS]
```

**Options**:

* `--ann PATH`: File with ASAP papers with extracted backgrounds and targets.  [required]
* `--context PATH`: File with ASAP papers with classified contexts.  [required]
* `--output PATH`: Full graph as a JSON file.  [required]
* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--help`: Show this message and exit.

## `peter citations`

Create citations graph.

**Usage**:

```console
$ peter citations [OPTIONS]
```

**Options**:

* `--asap PATH`: File with ASAP papers with references with full S2 data and classified contexts.  [required]
* `--output PATH`: Citation graph as a JSON file.  [required]
* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--help`: Show this message and exit.

## `peter query`

Demonstrate the graph by querying it to get polarised related papers.

**Usage**:

```console
$ peter query [OPTIONS]
```

**Options**:

* `--graph PATH`: Path to full graph file.  [required]
* `--asap-ann PATH`: File with ASAP papers with extracted backgrounds and targets.  [required]
* `--titles TEXT`: Title of the paper to test query. If absent, use an arbitrary paper.
* `-n, --num-papers INTEGER`: Number of papers to query if --title isn't given  [default: 1]
* `--help`: Show this message and exit.

## `peter semantic`

Create semantic graph.

**Usage**:

```console
$ peter semantic [OPTIONS]
```

**Options**:

* `--asap PATH`: File with ASAP papers with extracted backgrounds and targets.  [required]
* `--output PATH`: Semantic graph as a JSON file.  [required]
* `--model TEXT`: SentenceTransformer model to use.  [default: all-mpnet-base-v2]
* `--help`: Show this message and exit.
