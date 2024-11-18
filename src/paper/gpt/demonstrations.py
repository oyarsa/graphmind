"""Create demonstrations for few-shot prompting."""

from __future__ import annotations

import asyncio
from enum import StrEnum
import random
from pathlib import Path
from typing import Annotated

import aiohttp
import click
import typer
from pydantic import BaseModel, ConfigDict, TypeAdapter

from paper.gpt import annotate_paper as ann
from paper.gpt import evaluate_paper as eval
from paper.gpt.model import Paper
from paper.util.serde import save_data

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    help=__doc__,
)


@app.command(short_help="Full-paper evaluation.", no_args_is_help=True)
def eval_full(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON with paper data (asap_filtered.json)")
    ],
    output_file: Annotated[
        Path, typer.Argument(help="Path to output JSON file with the demonstrations")
    ],
    num_entries: Annotated[
        int, typer.Option("--entries", "-n", help="Number of entries for each type")
    ] = 10,
    seed: int = 0,
) -> None:
    """Create demonstrations for few-shot prompting of full-paper evaluation.

    Takes an even number of entries for positive and negative approval decisions, finds
    the review with lowest/highest reviews and uses it as the demonstration for the
    rationale.

    The input file is the output of the ASAP pipeline (asap_filtered.json).
    The output is a file with the paper title, abstract, main text, approval decision and
    the chosen rationale with its rating.
    """
    random.seed(seed)

    papers = TypeAdapter(list[Paper]).validate_json(input_file.read_bytes())

    papers_positive = random.sample([p for p in papers if p.approval], num_entries)
    papers_negative = random.sample([p for p in papers if not p.approval], num_entries)

    demonstrations = [
        new_eval_full_demonstration(paper, eval.DemonstrationType.POSITIVE)
        for paper in papers_positive
    ] + [
        new_eval_full_demonstration(paper, eval.DemonstrationType.NEGATIVE)
        for paper in papers_negative
    ]
    output_file.write_bytes(
        TypeAdapter(list[eval.Demonstration]).dump_json(demonstrations, indent=2)
    )


def new_eval_full_demonstration(
    paper: Paper, type_: eval.DemonstrationType
) -> eval.Demonstration:
    chosen_func = min if type_ is eval.DemonstrationType.NEGATIVE else max
    chosen = chosen_func(paper.reviews, key=lambda x: x.rating)

    return eval.Demonstration(
        title=paper.title,
        abstract=paper.abstract,
        text=paper.main_text(),
        approval=paper.approval,
        rationale=chosen.rationale,
        rating=chosen.rating,
        type=type_,
    )


class CSAbstructEntry(BaseModel):
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


if __name__ == "__main__":
    app()
