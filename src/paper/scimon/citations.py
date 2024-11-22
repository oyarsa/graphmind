"""Filter paper citation graph by title similarity, keeping top K only.

Calculates title sentence embedding for the ASAP main paper and each S2 reference using
a SentenceTransformer, then keep the top K most similar per paper. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `external_data.semantic_scholar.construct_daset`: the file
`asap_with_s2_references.json` of type `s2.ASAPWithFullS2`. The similarity is calculated
between the ASAP `title` and the S2 `title_query`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Self

import typer
from tqdm import tqdm

import paper.external_data.semantic_scholar.model as s2
from paper.scimon import embedding as emb
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
        s2_similarities = emb.similarities(asap_embedding, s2_embeddings)

        s2_top_k = sorted(
            zip(asap_paper.references, s2_similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        output.append(
            asap_paper.model_copy(update={"references": [ref for ref, _ in s2_top_k]})
        )

    return output


class TitleEncoder:
    def __init__(self, model_name: str) -> None:
        self._encoder = emb.Encoder(model_name)
        self._encoder.__enter__()

    def encode(self, title_raw: str) -> emb.Vector:
        """Clean and encode title as a vector.

        Title is cleaned-up with `s2.clean_title` first.
        """
        return self._encoder.encode(s2.clean_title(title_raw))

    def encode_multi(self, titles_raw: Iterable[str]) -> emb.Matrix:
        """Clean and parallel encode multiple titles as vectors.

        Titles are cleaned-up with `s2.clean_title` first.
        """
        return self._encoder.encode_multi(
            [s2.clean_title(title_raw) for title_raw in titles_raw]
        )

    def __enter__(self) -> Self:
        """Start encoder's context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close encoder's context."""
        return self._encoder.__exit__(*args)


if __name__ == "__main__":
    app()
