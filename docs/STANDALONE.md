# Standalone novelty evaluation

Evaluate the novelty of a paper (or many papers) by its arXiv title — as a
CLI command, or as a library from another Python project. Both interfaces do
the same work: resolve the title on Semantic Scholar and arXiv, parse its
LaTeX source, pull references and recommended/related papers, and run the
full PETER pipeline to produce a novelty evaluation.

## Prerequisites

- Python 3.12+
- `SEMANTIC_SCHOLAR_API_KEY` (required)
- `OPENAI_API_KEY` or `GEMINI_API_KEY` (at least one, depending on which
  `--llm-model`/`llm_model` you use)

Set these as real environment variables, or put them in a `.env` file in your
working directory (see `.env.example` in this repo for the full list of
variables used elsewhere in the project — only the three above matter for
this feature).

> [!NOTE]
> `pandoc` (needed for LaTeX parsing) is installed automatically as a Python
> dependency (`pypandoc-binary`) — no separate system install required.

## CLI usage

Install dependencies and the `graphmind` command:

```console
$ just sync          # or: uv sync --extra cpu --extra baselines
```

Evaluate a single paper:

```console
$ uv run graphmind single "Attention Is All You Need"
```

See `uv run graphmind single --help` for all options (model choice, number
of references/recommendations, output file, etc.).

Evaluate a batch of papers from a JSON file listing their titles:

```console
$ cat titles.json
["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"]

$ uv run graphmind single-batch titles.json --output out/
```

This writes one report per paper to `out/`, plus `out/failures.json` for any
titles that couldn't be resolved (not found on Semantic Scholar/arXiv, LaTeX
parse errors, etc.) — one bad title doesn't abort the rest of the batch.
Semantic Scholar lookups are cached and shared across the whole batch (in
`out/s2_cache.json` by default), so references or recommended papers common
to multiple titles are only fetched once. See `uv run graphmind single-batch
--help` for all options.

## Library usage

This package isn't published on PyPI (the name `paper` there belongs to an
unrelated project), so install it directly from the git repository:

```console
$ uv add "graphmind[cpu] @ git+https://github.com/oyarsa/graphmind.git"
```

> [!IMPORTANT]
> Use `uv`, and include the `cpu` (or `cuda`) extra, even though the library
> API itself doesn't need PyTorch. `sentence-transformers` is a core
> dependency (used elsewhere in the package) and pulls in `torch` regardless
> of which extra you pick — but *which* `torch` build gets resolved depends
> on the extra. This project pins CPU/CUDA-specific wheels via `uv`-only
> configuration (`[tool.uv.sources]`), which plain `pip` can't see: `pip
> install "graphmind[cpu] @ ..."` will still silently resolve a generic
> (often CUDA-bundled) `torch` build from PyPI, wasting a large, unnecessary
> download if you don't have a GPU. Only `uv add "graphmind[cpu] @ ..."`
> actually resolves the properly-sized CPU-only wheel. The `baselines` extra
> is never needed for library use — nothing in `evaluate_paper`/
> `evaluate_papers` touches it.

Then, from `paper.single_paper` (the distributed package is named
`graphmind`, but the importable module is still `paper` — see note below):

```python
import asyncio

from paper.single_paper import evaluate_paper, evaluate_papers, EvaluationError

async def main():
    # One paper — raises ValueError/RuntimeError on failure.
    result = await evaluate_paper("Attention Is All You Need")
    print(result.result.paper.label, result.result.paper.rationale_pred)

    # Many papers, concurrently, sharing one Semantic Scholar cache.
    evaluations = await evaluate_papers([
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    ])
    for evaluation in evaluations:
        if isinstance(evaluation.result, EvaluationError):
            print(f"{evaluation.title}: failed — {evaluation.result.message}")
        else:
            print(f"{evaluation.title}: label={evaluation.result.result.paper.label}")

asyncio.run(main())
```

`evaluate_paper`/`evaluate_papers` both take a `PaperEvaluationConfig` for
tuning pipeline parameters (model, number of references/recommendations,
prompt keys, etc.) — defaults match the CLI. `evaluate_papers` also takes a
`cache_path` to persist the Semantic Scholar cache across calls/runs (like
the CLI's batch command), and a `concurrency` limit.

> [!NOTE]
> The distributed package name is `graphmind`, but `src/paper` (the
> importable module path) wasn't renamed along with it — `import
> paper.single_paper` is correct, not `import graphmind.single_paper`.

## Troubleshooting

- **"Paper not found on Semantic Scholar" despite a valid, freshly-set API
  key**: check whether `SEMANTIC_SCHOLAR_API_KEY` is already set directly in
  your shell (`echo $SEMANTIC_SCHOLAR_API_KEY`). `.env` loading doesn't
  override existing environment variables by default, so a stale key
  exported in your shell will silently shadow a correct one in `.env`. Run
  `unset SEMANTIC_SCHOLAR_API_KEY` (or fix the shell-level value) if this
  happens.
