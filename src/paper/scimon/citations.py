"""Create citations graph with the top K titles from cited papers.

Calculates title sentence embedding for the ASAP main paper and each S2 reference using
a SentenceTransformer, then keep the top K most similar per paper. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `external_data.semantic_scholar.construct_daset`: the file
`asap_with_s2_references.json` of type `s2.ASAPWithFullS2`. The similarity is calculated
between the ASAP `title` and the S2 `title_query`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated

import typer
from openai import BaseModel
from pydantic import ConfigDict, computed_field
from tqdm import tqdm

import paper.external_data.semantic_scholar.model as s2
from paper.scimon import embedding as emb
from paper.util import display_params
from paper.util.serde import Record, load_data

logger = logging.getLogger(__name__)

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
    logger.info(display_params())

    asap_papers = load_data(input_file, s2.ASAPWithFullS2)

    with emb.Encoder(model_name) as encoder:
        graph = keep_top_k_titles(asap_papers, encoder, k)

    output_file.write_text(graph.model_dump_json(indent=2))


def keep_top_k_titles(
    asap_papers: Iterable[s2.ASAPWithFullS2], encoder: emb.Encoder, k: int
) -> Graph:
    """For each ASAP paper, find top K reference titles embedding similarity.

    Cleans up the titles with `s2.clean_title`, then compares the ASAP `title` with the
    S2 `title_query`.
    """
    paperid_to_cited: dict[int, list[Citation]] = {}

    for asap_paper in tqdm(asap_papers, desc="Processing ASAP papers"):
        asap_embedding = encode_title(encoder, asap_paper.title)

        s2_embeddings = encode_titles(
            encoder, (s2.title_query for s2 in asap_paper.references)
        )
        s2_similarities = emb.similarities(asap_embedding, s2_embeddings)

        s2_top_k = sorted(
            zip(asap_paper.references, s2_similarities),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        paperid_to_cited[asap_paper.id] = [
            Citation(
                title_s2=paper.title,
                title_asap=paper.title_query,
                paper_id=paper.paper_id,
            )
            for paper, _ in s2_top_k
        ]

    return Graph(edge_list=paperid_to_cited)


class Graph(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_list: Mapping[int, Sequence[Citation]]
    """Mapping of ASAP paper `id` to list of cited papers."""

    @computed_field
    @property
    def nodes(self) -> Sequence[int]:
        return sorted(self.edge_list)

    def query(self, paper_id: int) -> Sequence[Citation]:
        return self.edge_list[paper_id]


class Citation(Record):
    title_s2: str
    title_asap: str
    paper_id: str

    @property
    def id(self) -> str:
        return self.paper_id


def encode_title(encoder: emb.Encoder, title_raw: str) -> emb.Vector:
    """Clean and encode title as a vector.

    Title is cleaned-up with `s2.clean_title` first.
    """
    return encoder.encode(s2.clean_title(title_raw))


def encode_titles(encoder: emb.Encoder, titles_raw: Iterable[str]) -> emb.Matrix:
    """Clean and parallel encode multiple titles as vectors.

    Titles are cleaned-up with `s2.clean_title` first.
    """
    return encoder.encode_multi([s2.clean_title(title_raw) for title_raw in titles_raw])


if __name__ == "__main__":
    app()
