# external_data: Interact with external APIs

Uses CrossRef and Semantic Scholar API to search papers via title similarity. CrossRef
doesn't work well, but Semantic Scholar is a bit better.

There are also scripts to manage the S2ORC datasets through AWS S3 and Athena. This
allows us to perform queries over large amounts of data, with some complexity associated
with having the files on the cloud instead of locally.

## Tool descriptions

- [`crossref.py`](crossref.py): Download paper information from the CrossRef API.
- [`s3_athena_tables.py`](s3_athena_tables.py): Create Athena tables for S2ORC datasets:
  `papers`, `abstracts` and `papers_enriched`.
- [`s3_datasets.py`](s3_datasets.py): Stream S2ORC datasets to AWS S3, or upload
  already-downloaded data.
- [`semantic_scholar_areas.py`](semantic_scholar_areas.py): Download papers belonging to
  ICLR primary areas from the Semantic Scholar API.
- [`semantic_scholar_info.py`](semantic_scholar_info.py): Download paper information
  from the Semantic Scholar API.
- [`semantic_scholar_model.py`](semantic_scholar_model.py): Data models for types
  returned by the Semantic Scholar API.
- [`semantic_scholar_recommended.py`](semantic_scholar_recommended.py): Download paper
  recommendations for PeerRead papers from the Semantic Scholar API.
- [`semantic_scholar_references.py`](semantic_scholar_references.py): Download
  information for papers referenced in PeerRead using the Semantic Scholar API.

## Notes

We combine the outputs of `semantic_scholar_recommended` and `semantic_scholar_areas`
into a single file: `semantic_scholar_related.json`. We want to build a dataset of
papers related to our inputs, and `areas` and `recommended` are different approaches to
that that we can combine. Some statistics:

- `papers_recommended.json`: 95 359 unique papers
- `papers_areas.json`: 50 372 unique papers
- `papers_related.json`: 145 731 total papers, 141 487 unique.
