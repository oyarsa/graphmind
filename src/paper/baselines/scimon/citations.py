"""Create citations graph with the reference papers sorted by title similarity.

Calculates title sentence embedding for the PeerRead main paper and each S2 reference
using a SentenceTransformer, then keeping a sorted list by similarity. At query time,
the user can specify the number of most similar papers to retrieve. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `semantic_scholar.construct_daset`: the file
`peerread_with_s2_references.json` of type `peerread.PaperWithS2Refs`. The similarity is
calculated between the PeerRead `title` and the S2 `title_peer`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Protocol, Self

import typer
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

import paper.semantic_scholar as s2
from paper import embedding as emb
from paper.util import display_params
from paper.util.serde import Record, load_data, save_data

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
        typer.Argument(
            help="File with PeerRead papers with references with full S2 data."
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(
            help="File with PeerRead papers with top K references with full S2 data."
        ),
    ],
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    peerread_papers = load_data(input_file, s2.PaperWithS2Refs)

    encoder = emb.Encoder(model_name)
    graph = Graph.from_papers(encoder, peerread_papers)

    save_data(output_file, graph)


class MainPaper(Protocol):
    """Main paper with title, id and references."""

    @property
    def title(self) -> str:
        """Title of the paper in the main dataset."""
        ...

    @property
    def id(self) -> str:
        """Unique identifier of the paper."""
        ...

    @property
    def references(self) -> Iterable[S2Reference]:
        """Other papers cited by this paper."""
        ...


class S2Reference(Protocol):
    """S2-shaped paper referenced in an main paper."""

    @property
    def paper_id(self) -> str:
        """Unique identifier of the paper."""
        ...

    @property
    def title(self) -> str:
        """Paper title in the S2 API."""
        ...

    @property
    def title_peer(self) -> str:
        """Title of the paper in the PeerRead dataset used to query S2."""
        ...


class Graph(BaseModel):
    """Citation graph that connects main paper titles/ids with cited paper titles.

    We retrieve the top K titles by main and reference titles. We embed all of them, and
    the K is determined at query time.
    """

    model_config = ConfigDict(frozen=True)

    title_to_id: Mapping[str, str]
    """Mapping from paper `title` to its `id`.

    This is meant for testing; use the paper_id directly for querying, as there can be
    duplicates by `title`, but the `id` is always unique.
    """
    id_to_cited: Mapping[str, Sequence[Citation]]
    """Mapping of main paper `id` to list of cited papers."""

    @classmethod
    def from_papers(
        cls,
        encoder: emb.Encoder,
        peerread_papers: Iterable[MainPaper],
        *,
        progress: bool = False,
    ) -> Self:
        """For each main paper, sort cited papers by title similarity.

        Cleans up the titles with `s2.clean_title`, then compares the main paper `title`
        with the S2 `title_peer`.
        """
        title_to_id: dict[str, str] = {}
        id_to_cited: dict[str, list[Citation]] = {}

        if progress:
            peerread_papers = tqdm(peerread_papers, desc="Citation graph")

        logger.debug("Processing papers.")
        for peer_paper in peerread_papers:
            title_to_id[peer_paper.title] = peer_paper.id
            peer_embedding = encoder.encode(s2.clean_title(peer_paper.title))
            titles = [s2.clean_title(r.title_peer) for r in peer_paper.references]

            s2_embeddings = encoder.batch_encode(titles)
            s2_similarities = emb.similarities(peer_embedding, s2_embeddings)

            id_to_cited[peer_paper.id] = [
                Citation(title=paper.title, paper_id=paper.paper_id, score=score)
                for paper, score in sorted(
                    zip(peer_paper.references, s2_similarities),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

        logger.debug("Done.")

        return cls(id_to_cited=id_to_cited, title_to_id=title_to_id)

    def query_title(self, title: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from a main paper `title`.

        Note: prefer `query` for actual usage. See `title_to_id`.
        """
        return self.query(self.title_to_id[title], k)

    def query(self, paper_id: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from a main paper `id`."""
        return QueryResult(citations=self.id_to_cited[paper_id][:k])

    @property
    def nodes(self) -> Sequence[str]:
        """Nodes are paper ids."""
        return sorted(self.id_to_cited)


class Citation(Record):
    """Encoded citation with the S2 paper title, ID and similarity score."""

    title: str
    paper_id: str
    score: float

    @property
    def id(self) -> str:
        """Identify the S2 by its paper_id from the API."""
        return self.paper_id


class QueryResult(BaseModel):
    """Result of the citation query: the top K cited papers."""

    model_config = ConfigDict(frozen=True)

    citations: Sequence[Citation]


if __name__ == "__main__":
    app()
