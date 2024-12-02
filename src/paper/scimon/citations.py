"""Create citations graph with the reference papers sorted by title similarity.

Calculates title sentence embedding for the ASAP main paper and each S2 reference using
a SentenceTransformer, then keeping a sorted list by similarity. At query time, the user
can specify the number of most similar papers to retrieve. SciMON uses the
`all-mpnet-base-v2` model and K = 5.

Takes as input the output of `semantic_scholar.construct_daset`: the file
`asap_with_s2_references.json` of type `s2.ASAPWithFullS2`. The similarity is calculated
between the ASAP `title` and the S2 `title_query`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Self

import typer
from openai import BaseModel
from pydantic import ConfigDict

import paper.semantic_scholar as s2
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
    model_name: Annotated[
        str, typer.Option("--model", help="SentenceTransformer model to use.")
    ] = "all-mpnet-base-v2",
) -> None:
    """Create citations graph with the reference papers sorted by title similarity."""
    logger.info(display_params())

    asap_papers = load_data(input_file, s2.ASAPWithFullS2)

    encoder = emb.Encoder(model_name)
    graph = Graph.from_papers(encoder, asap_papers)

    output_file.write_text(graph.model_dump_json(indent=2))


class Graph(BaseModel):
    """Citation graph that connects main ASAP titles/ids with cited paper titles.

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
    """Mapping of ASAP paper `id` to list of cited papers."""

    @classmethod
    def from_papers(
        cls, encoder: emb.Encoder, asap_papers: Iterable[s2.ASAPWithFullS2]
    ) -> Self:
        """For each ASAP paper, sort cited papers by title similarity.

        Cleans up the titles with `s2.clean_title`, then compares the ASAP `title` with
        the S2 `title_query`.
        """
        title_to_id: dict[str, str] = {}
        id_to_cited: dict[str, list[Citation]] = {}

        logger.debug("Processing papers.")
        for asap_paper in asap_papers:
            title_to_id[asap_paper.title] = asap_paper.id
            asap_embedding = encoder.encode(s2.clean_title(asap_paper.title))

            s2_embeddings = encoder.encode(
                [s2.clean_title(r.title_query) for r in asap_paper.references]
            )
            s2_similarities = emb.similarities(asap_embedding, s2_embeddings)

            id_to_cited[asap_paper.id] = [
                Citation(title=paper.title, paper_id=paper.paper_id, score=score)
                for paper, score in sorted(
                    zip(asap_paper.references, s2_similarities),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

        logger.debug("Done.")

        return cls(id_to_cited=id_to_cited, title_to_id=title_to_id)

    def query_title(self, title: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from an ASAP paper `title`.

        Note: prefer `query` for actual usage. See `title_to_id`.
        """
        return self.query(self.title_to_id[title], k)

    def query(self, paper_id: str, k: int) -> QueryResult:
        """Get top `k` cited papers by title similarity from an ASAP paper `id`."""
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
