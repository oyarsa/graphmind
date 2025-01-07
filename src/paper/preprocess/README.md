# preprocess: Transform datasets

The main dataset we use is [ASAP](https://github.com/neulab/ReviewAdvisor#dataset). It
contains peer reviews, ratings and approval decisions for papers in ICLR and NeurIPS. We
use only the ICLR subset because it contains all the information we need, while NeurIPS
is missing some.

There is also the S2ORC dataset, which can be used to retrieve information from a large
collection of papers from [Semantic Scholar](https://api.semanticscholar.org/api-docs/datasets).
We don't actually use it anymore because we're querying the Semantic Scholar API
directly (see `paper.semantic_scholar`), but the processing code is still there.

The primary way to process the data is through the `preprocess` command:

```console
$ uv run preprocess
```

For full documentation on how to use the tools, see [`CLI.md`](./CLI.md).
