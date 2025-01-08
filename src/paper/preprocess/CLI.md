# `s2orc`

Run S2ORC preprocessing pipeline

**Usage**:

```console
$ s2orc [OPTIONS] PROCESSED_DIR OUTPUT_PATH DATASET_PATH
```

**Arguments**:

* `PROCESSED_DIR`: Path to save the downloaded dataset.  [required]
* `OUTPUT_PATH`: Path to save the S2 extracted files (JSON.GZ).  [required]
* `DATASET_PATH`: Path to save the output (processed and filtered - ACL only) files  [required]

**Options**:

* `--file-limit INTEGER`: Limit the number of files to download. If not provided, download all.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.
