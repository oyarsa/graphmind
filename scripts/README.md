# scripts: Independent scripts

This directory is a collection of (potentially) throwaway scripts I write over the
course of development that could be useful again in the future, so I don't want to just
delete them.

Every script will include a docstring describing what they do and their context, but I
make no promises about structure. However, they should still pass the check task,
including ruff checks and pyright. It's acceptable that these scripts use `pyright:
basic`, although they should still have _some_ form of typing.

## Organisation

### Subdirectories

- [`tools`](tools/README.md): Various helper tools.
- [`experiments`](experiments/README.md): Run experiments and analyse results.
- [`transform`](transform/README.md): Transform data files.

### Other

These are scripts that don't fit well into the categories above:

- [`asap_avg_unique_ref_mentions.py`](asap_avg_unique_ref_mentions.py): Calculate total
  and average number of unique reference contexts per paper in ASAP.
- [`cmp_ratings_approval.py`](cmp_ratings_approval.py): Compare classification metrics
  between ratings evaluation strategies and approval.
- [`count_context.py`](count_context.py): Count number of context items in data file.
  Show polarity frequencies, if available.
