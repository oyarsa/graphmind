# `peerread`

Run PeerRead dataset commands.

**Usage**:

```console
$ peerread [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `download`: Download PeerRead dataset.
* `preprocess`: Run PeerRead preprocessing pipeline.

## `peerread download`

Download PeerRead dataset.

**Usage**:

```console
$ peerread download [OPTIONS] OUTPUT_DIR
```

**Arguments**:

* `OUTPUT_DIR`: Path to save the PeerRead data.  [required]

**Options**:

* `--help`: Show this message and exit.

## `peerread preprocess`

Run PeerRead preprocessing pipeline.

**Usage**:

```console
$ peerread preprocess [OPTIONS] PATH OUTPUT_FILE
```

**Arguments**:

* `PATH`: Path to directories containing files to merge.  [required]
* `OUTPUT_FILE`: Output merged JSON file.  [required]

**Options**:

* `-n, --max-papers INTEGER`: Limit on the number of papers to process.
* `--help`: Show this message and exit.
