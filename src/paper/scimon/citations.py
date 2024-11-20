"""Filter paper citation graph by title similarity, keeping top K only.

Calculates title sentence embedding for the ASAP main paper and each S2 reference using
a SentenceTransformer, then keep the top K most similar per paper. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `external_data.semantic_scholar.construct_daset`: the file
`asap_with_s2_references.json` of type `s2.ASAPWithFullS2`. The similarity is calculated
between the ASAP `title` and the S2 `title_query`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import typer
from tqdm import tqdm

import paper.external_data.semantic_scholar.model as s2
from paper.util import display_params
from paper.util.serde import load_data, save_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__, no_args_is_help=True)
def main(
    input_file: Annotated[
        Path,
        typer.Argument(help="File with ASAP papers with references with full S2 data."),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(
            help="File with ASAP papers with top K references with full S2 data."
        ),
    ],
    k: Annotated[int, typer.Option(help="Top K items to keep.")] = 5,
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    print(display_params())

    asap_papers = load_data(input_file, s2.ASAPWithFullS2)

    with TitleEncoder(model_name) as encoder:
        asap_topk_refs = _keep_top_k_titles(asap_papers, encoder, k)

    save_data(output_file, asap_topk_refs)


def _keep_top_k_titles(
    asap_papers: Iterable[s2.ASAPWithFullS2], encoder: TitleEncoder, k: int
) -> list[s2.ASAPWithFullS2]:
    """For each ASAP paper, keep top K references by title embedding similarity.

    Cleans up the titles with `s2.clean_title`, then compares the ASAP `title` with the
    S2 `title_query`.

    The output has the same format as the input, except with fewer (top K) references.
    """
    output: list[s2.ASAPWithFullS2] = []

    for asap_paper in tqdm(asap_papers, desc="Processing ASAP papers"):
        asap_embedding = encoder.encode(asap_paper.title)

        s2_embeddings = encoder.encode_multi(
            s2.title_query for s2 in asap_paper.references
        )
        s2_similarities = _similarities(asap_embedding, s2_embeddings)

        s2_top_k = sorted(
            zip(asap_paper.references, s2_similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        output.append(
            asap_paper.model_copy(update={"references": [ref for ref, _ in s2_top_k]})
        )

    return output


type TitleEmbedding = npt.NDArray[np.float32]
type TitleEmbeddingMatrix = npt.NDArray[np.float32]
type SentenceTransformerPool = dict[Literal["input", "output", "processes"], Any]


class TitleEncoder:
    def __init__(self, model_name: str) -> None:
        # `sentence_transformers` has a bug where they don't clean up their semaphores
        # properly, so we suppress this.
        _hard_suppress_warning("multiprocessing.resource_tracker", "UserWarning")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

        # Used for parallel processing.
        self._pool = self._model.start_multi_process_pool()

    def encode(self, title_raw: str) -> TitleEmbedding:
        """Clean and encode title as a vector.

        Title is cleaned-up with `s2.clean_title` first.
        """
        title_clean = s2.clean_title(title_raw)
        return cast(TitleEmbedding, self._model.encode(title_clean))  # type: ignore

    def encode_multi(self, titles_raw: Iterable[str]) -> TitleEmbeddingMatrix:
        """Clean and parallel encode multiple titles as vectors.

        Titles are cleaned-up with `s2.clean_title` first.
        """
        titles_clean = [s2.clean_title(title_raw) for title_raw in titles_raw]

        embeddings = self._model.encode_multi_process(titles_clean, self._pool)  # type: ignore
        return cast(TitleEmbeddingMatrix, embeddings)

    def __enter__(self) -> Self:
        """Start multiprocessing pool."""
        return self

    def __exit__(self, *_) -> None:
        """Close multiprocessing pool."""
        self._model.stop_multi_process_pool(self._pool)


def _similarities(
    vector: TitleEmbedding, matrix: TitleEmbeddingMatrix
) -> npt.NDArray[np.float32]:
    """Calculate cosine similarity between a `vector` and a `matrix`.

    The `vector` will have shape (dim,), and matrices (N, dim), where N is the number of
    entries. The output should be a vector of floats with N elements.

    Raises:
        ValueError: if vector or matrix have the wrong shape.
    """
    # Preconditions
    if vector.ndim != 1:
        raise ValueError("vector must be 1D")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if vector.shape[0] != matrix.shape[1]:
        raise ValueError("vector and matrix dimensions must be compatible")

    dot_product = np.dot(matrix, vector)

    # Compute the L2 norms (magnitudes)
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    denominator = (vector_norm * matrix_norms) + epsilon

    similarities = (dot_product / denominator).astype(np.float32)

    # Postconditions
    assert similarities.ndim == 1, "similarities should be 1D"
    assert (
        similarities.shape[0] == matrix.shape[0]
    ), "similarities must have as many elements as matrix rows"

    return similarities


def _hard_suppress_warning(name: str, type: str) -> None:
    """Suppress warning by adding it to `PYTHONWARNINGS`."""
    old_warnings = os.environ.get("PYTHONWARNINGS")
    new_filter = f"ignore::{type}:{name}"
    os.environ["PYTHONWARNINGS"] = (
        f"{old_warnings},{new_filter}" if old_warnings else new_filter
    )


if __name__ == "__main__":
    app()
