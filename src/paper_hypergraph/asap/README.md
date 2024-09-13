# ASAP-Review processing

Merge the files from the papers in the ASAP-Review dataset into a single file. Note that
only entries that have ratings in their reviews will be used.

## Usage

- Download the dataset from [Google Drive](https://drive.usercontent.google.com/download?id=1nJdljy468roUcKLbVwWUhMs7teirah75&export=download&authuser=0).
- Extract it to the `data` folder.
- Run the merge script on the ICLR dataset.
```bash
$ python merge.py ICLR* -o output.json
```

The NIPS papers don't have ratings in their reviews.

See README.original.md for the original README file for the dataset.
