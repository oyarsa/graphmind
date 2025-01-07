# `preprocess`

Run different preprocessing pipelines.

**Usage**:

```console
$ preprocess [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `asap`: Run ASAP-Review preprocessing pipeline
* `s2orc`: Run S2ORC preprocessing pipeline

## `preprocess asap`

Run ASAP-Review preprocessing pipeline

**Usage**:

```console
$ preprocess asap [OPTIONS] PAPERS_PATH OUTPUT_PATH [MAX_PAPERS]
```

**Arguments**:

* `PAPERS_PATH`: Path to input directory containing raw ASAP files.  [required]
* `OUTPUT_PATH`: Path to output directory for processed files.  [required]
* `[MAX_PAPERS]`: Limit on the number of papers to process.

**Options**:

* `--help`: Show this message and exit.

## `preprocess s2orc`

Run S2ORC preprocessing pipeline

**Usage**:

```console
$ preprocess s2orc [OPTIONS] PROCESSED_DIR OUTPUT_PATH DATASET_PATH
```

**Arguments**:

* `PROCESSED_DIR`: Path to save the downloaded dataset.  [required]
* `OUTPUT_PATH`: Path to save the S2 extracted files (JSON.GZ).  [required]
* `DATASET_PATH`: Path to save the output (processed and filtered - ACL only) files  [required]

**Options**:

* `--file-limit INTEGER`: Limit the number of files to download. If not provided, download all.
* `--help`: Show this message and exit.
