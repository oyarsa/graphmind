"""Create demonstrations for few-shot prompting."""

from __future__ import annotations

import asyncio
import random
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import aiohttp
import typer
from pydantic import BaseModel, ConfigDict

from paper import peerread as pr
from paper.gpt import annotate_paper as ann
from paper.gpt import evaluate_paper as eval
from paper.util import groupby, shuffled
from paper.util.serde import load_data, save_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(short_help="Full-paper evaluation.", no_args_is_help=True)
def eval_sans(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON with paper data (peerread_merged.json)")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file with the demonstrations")
    ],
    num_entries: Annotated[
        int, typer.Option("--entries", "-n", help="Number of entries to sample")
    ] = 10,
    seed: int = 0,
) -> None:
    """Create demonstrations for few-shot prompting of full-paper evaluation.

    Takes a number of entries to sample and returns the chosen papers with their rating
    and rationale.

    The input file is the output of the PeerRead pipeline (peerread_merged.json).
    The output is a file with the paper title, abstract, main text, novelty rating and
    rationale.
    """
    random.seed(seed)

    papers = load_data(input_file, pr.Paper)
    papers_sample = random.sample(papers, num_entries)

    demonstrations = [new_eval_sans_demonstration(paper) for paper in papers_sample]
    save_data(output_file, demonstrations)


def new_eval_sans_demonstration(paper: pr.Paper) -> eval.Demonstration:
    """Construct demonstration for sans paper (main paper only) evaluation."""
    return eval.Demonstration(
        title=paper.title,
        abstract=paper.abstract,
        text=paper.main_text,
        rationale=paper.rationale,
        rating=paper.rating,
    )


class CSAbstructEntry(BaseModel):
    """Entry from the CSAbstruct dataset used for abstract sentence classification."""

    model_config = ConfigDict(frozen=True)

    sentences: list[str]
    labels: list[str]


async def download_abstracts(url: str) -> str:
    """Download CSAbstruct JSONL data."""
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        response.raise_for_status()
        return await response.text()


def new_abstract_demonstration(entry: CSAbstructEntry) -> ann.AbstractDemonstration:
    """Create an abstract demonstration from a CSAbstructEntry.

    Combines `sentences` into a single `abstract` text, and separates `background` and
    `target` by their `labels`.
    """

    background_sentences: list[str] = []
    target_sentences: list[str] = []

    for sentence, label in zip(entry.sentences, entry.labels):
        if label == "background":
            background_sentences.append(sentence)
        else:
            target_sentences.append(sentence)

    return ann.AbstractDemonstration(
        abstract=" ".join(entry.sentences).strip(),
        background=" ".join(background_sentences).strip(),
        target=" ".join(target_sentences).strip(),
    )


# URL is live as of 2024-11-19
# Source: https://github.com/allenai/sequential_sentence_classification (Apache-2.0).
CSABSTRUCT_BASE_URL = "https://raw.githubusercontent.com/allenai/sequential_sentence_classification/cf5ad6c663550dd8203f148cd703768d9ee86ff4/data/CSAbstruct/{split}.jsonl"


class CSAbstructSplit(StrEnum):
    """Which data split of the CSAbstruct dataset to use to build demonstrations."""

    DEV = "dev"
    TRAIN = "train"
    TEST = "test"


async def process_abstracts(
    output_path: Path, num_entries: int, base_url: str, split: CSAbstructSplit
) -> None:
    """Download, process, and save CS abstract demonstrations."""
    url = base_url.format(split=split.value)
    content = await download_abstracts(url)

    entries = [
        CSAbstructEntry.model_validate_json(line) for line in content.splitlines()
    ][:num_entries]
    demonstrations = [new_abstract_demonstration(entry) for entry in entries]

    save_data(output_path, demonstrations)


@app.command(short_help="Abstract classification.", no_args_is_help=True)
def abstract(
    output_path: Annotated[
        Path, typer.Argument(help="Path to save the demonstrations.")
    ],
    entries: Annotated[
        int,
        typer.Option("--entries", "-n", help="Number of demonstrations to process."),
    ] = 10,
    base_url: Annotated[
        str, typer.Option("--url", help="URL of the JSONL file to process.")
    ] = CSABSTRUCT_BASE_URL,
    split: Annotated[
        CSAbstructSplit, typer.Option(help="Data split to use for demonstrations.")
    ] = CSAbstructSplit.DEV,
) -> None:
    """Create demonstrations for few-shot abstract classification.

    Downloads data from the CSAbstruct dataset as a JSONL file, then processes to our
    format of abstract classification.
    """
    asyncio.run(process_abstracts(output_path, entries, base_url, split))


@app.command(short_help="Review evaluation.", no_args_is_help=True)
def eval_reviews(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON with paper data (peerread_merged.json)")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file with the demonstrations")
    ],
    num_entries: Annotated[
        int, typer.Option("--entries", "-n", help="Number of entries to sample")
    ] = 10,
    seed: int = 0,
) -> None:
    """Create demonstrations for few-shot prompting of review evaluation.

    Takes a number of entries to sample and returns the chosen papers with their rating
    and rationale. The number of entries must be a multiple of 5 since we'll have one
    demonstration with each rating.

    The input file is the output of the PeerRead pipeline (peerread_merged.json).
    The output is a file with the paper title, abstract, main text, novelty rating and
    rationale.
    """
    random.seed(seed)

    if num_entries % 5 != 0:
        raise ValueError("`num_entries` must be a multiple of 5.")

    num_each = num_entries // 5
    papers = load_data(input_file, pr.Paper)

    reviews = [(paper, review) for paper in papers for review in paper.reviews]
    reviews_grouped = groupby(reviews, lambda x: x[1].rating)

    reviews_chosen = {
        rating: shuffled(reviews)[:num_each]
        for rating, reviews in reviews_grouped.items()
    }
    reviews_final = [
        review for reviews in reviews_chosen.values() for review in reviews
    ]

    demonstrations = [
        new_review_evaluation(paper, review) for paper, review in reviews_final
    ]
    save_data(output_file, demonstrations)


def new_review_evaluation(
    paper: pr.Paper, review: pr.PaperReview
) -> eval.Demonstration:
    """Construct demonstration for review evaluation."""
    return eval.Demonstration(
        title=paper.title,
        abstract=paper.abstract,
        text=paper.main_text,
        rationale=review.rationale,
        rating=review.rating,
    )


@app.command(short_help="Novelty evaluation with binary label.", no_args_is_help=True)
def eval_binary(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON with paper data (peerread_merged.json)")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file with the demonstrations")
    ],
    num_entries: Annotated[
        int, typer.Option("--entries", "-n", help="Number of entries to sample")
    ] = 10,
    seed: int = 0,
) -> None:
    """Create demonstrations for few-shot prompting of novelty evaluation.

    Takes a number of entries to sample and returns the chosen papers with their rating
    and rationale. The number of entries should be even, and we'll pick an equal number
    of positive and negative reviews.

    The output is a file with the paper title, abstract, main text, novelty rating,
    novelty label (binary) and rationale.
    """
    random.seed(seed)

    data = load_data(input_file, pr.Paper)
    n = num_entries // 2

    pos = [x for x in data if x.label]
    neg = [x for x in data if not x.label]

    pos_sampled = random.sample(pos, n)
    neg_sampled = random.sample(neg, n)

    save_data(output_file, pos_sampled + neg_sampled)


if __name__ == "__main__":
    app()
