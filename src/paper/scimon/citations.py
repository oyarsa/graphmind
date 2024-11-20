"""Filter paper citation graph by title similarity, keeping top K only.

Calculates title sentence embedding for the ASAP main paper and each S2 reference using
SentenceTransformer, then keep the top K most similar per paper. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `external_data.semantic_scholar.construct_daset`, the file
`asap_with_s2_references.json` of type `s2.ASAPWithFullS2`. The similarity is calculated
between the ASAP `title` and the S2 `title_query`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Self, cast

import numpy as np
import numpy.typing as npt
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import paper.asap.model as asap
import paper.external_data.semantic_scholar.model as s2
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
    asap_papers = load_data(input_file, s2.ASAPWithFullS2)

    encoder = TitleEncoder(model_name)
    asap_topk_refs = _keep_top_k_titles(asap_papers, encoder, k)

    save_data(output_file, asap_topk_refs)


def _keep_top_k_titles(
    asap_papers: Iterable[s2.ASAPWithFullS2], encoder: TitleEncoder, k: int
) -> list[s2.ASAPWithFullS2]:
    """For each ASAP paper, keep top K references by title embedding similarity.

    Compares the ASAP `title` with the S2 `title_query`. Cleans up the titles with
    `s2.clean_title` first.

    The output has the same format as the input, except with fewer (top K) references.
    """
    output: list[s2.ASAPWithFullS2] = []

    for asap_paper in tqdm(asap_papers, desc="Processing ASAP papers"):
        asap_embedding = encoder.encode(asap_paper.title)

        s2_references = [
            S2PaperEncoded.from_(
                s2_ref,
                asap_paper.title,
                _similarity(asap_embedding, encoder.encode(s2_ref.title_query)),
            )
            for s2_ref in asap_paper.references
        ]

        s2_top_k = sorted(s2_references, key=lambda x: x.similarity, reverse=True)[:k]
        output.append(asap_paper.model_copy(update={"references": s2_top_k}))

    return output


class S2PaperEncoded(asap.S2Paper):
    """Same as the S2 referenced papers from ASAP, but with the source and similarity.

    Attributes:
        source_title: Title of the paper that cited this.
        similarity: Cosine similarity between the `source_title` and this `title_query`.
    """

    source_title: str
    similarity: float

    @classmethod
    def from_(cls, paper: asap.S2Paper, source_title: str, similarity: float) -> Self:
        return cls(
            title_query=paper.title_query,
            title=paper.title,
            paperId=paper.paper_id,
            corpusId=paper.corpus_id,
            url=paper.url,
            abstract=paper.abstract,
            year=paper.year,
            referenceCount=paper.reference_count,
            citationCount=paper.citation_count,
            influentialCitationCount=paper.influential_citation_count,
            tldr=paper.tldr,
            authors=paper.authors,
            source_title=source_title,
            similarity=similarity,
        )


type TitleEmbedding = npt.NDArray[np.float32]


class TitleEncoder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        # Cache of `title_clean` -> `embedding`.
        self._cache: dict[str, TitleEmbedding] = {}

    def encode(self, title_raw: str) -> TitleEmbedding:
        """Clean and encode title as a vector using a `SentenceTransformer`.

        Titles are cleaned-up with `s2.clean_title` first. Keeps a cache of vectors by
        the title name to avoid recomputing the same data.
        """
        title_clean = s2.clean_title(title_raw)
        if (embedding := self._cache.get(title_clean)) is not None:
            return embedding

        embedding = cast(TitleEmbedding, self._model.encode(title_clean))  # type: ignore
        self._cache[title_clean] = embedding
        return embedding


def _similarity(v1: TitleEmbedding, v2: TitleEmbedding) -> float:
    """Calculate cosine similarity between two NumPy vectors.

    They both must be 1-dimensional and have the same number of items.
    """
    # Ensure inputs are 1D
    assert v1.ndim == 1 and v2.ndim == 1, "Input arrays must be 1-dimensional"
    assert v1.shape[0] == v2.shape[0], "Input vectors must have same dimensionality"

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
if __name__ == "__main__":
    app()
