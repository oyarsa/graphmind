"""Match papers in `papers` dataset with `abstracts`.

Uses sqlite3 to build two tables: `abstracts`, with `(corpusid, abstract)`, and `papers`,
with `(corpusid, payload)`, where `payload` is the entire JSON item in string form.

After building the two tables (in batches so we don't run out of memory when reading
the input files), we (left) join them and build batched output files with the result.

While the output files are gzipped JSON Lines, the output files are JSON arrays. The
content is almost the same, with the addition of the `abstract` key, which can be null,
if there was no match.
"""

import gzip
import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TextIO

import typer
from tqdm import tqdm

ABSTRACT_BATCH_SIZE = 500_000
PAPER_BATCH_SIZE = 100_000
OUTPUT_BATCH_SIZE = 10_000

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    papers_dir: Annotated[
        Path,
        typer.Option(
            "--papers",
            help="Path to directory containing the papers dataset files.",
        ),
    ],
    abstracts_dir: Annotated[
        Path,
        typer.Option(
            "--abstracts",
            help="Path to directory containing the abstracts dataset files.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", help="Path to directory for output files.")
    ],
    db_path: Annotated[
        Path, typer.Option("--db", help="Path for SQLite database file.")
    ] = Path("paper_abstract.sqlite"),
    max_papers: Annotated[
        int | None,
        typer.Option(help="Maximum number of papers to go through. Defaults to all."),
    ] = None,
    max_abstracts: Annotated[
        int | None,
        typer.Option(
            help="Maximum number of abstracts to go through. Defaults to all."
        ),
    ] = None,
) -> None:
    """Match papers with abstracts using SQLite as intermediate storage."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with closing(setup_database(db_path)) as conn:
        load_abstracts(conn, abstracts_dir, max_abstracts)
        load_papers(conn, papers_dir, max_papers)
        process_matches(conn, output_dir)


@dataclass(frozen=True, kw_only=True)
class Table:
    name: str
    column: str


ABSTRACTS_TABLE = Table(name="abstracts", column="abstract")
PAPERS_TABLE = Table(name="papers", column="payload")


def setup_database(db_path: Path) -> sqlite3.Connection:
    """Create database tables for abstracts and papers."""
    conn = sqlite3.connect(db_path)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    for table in (ABSTRACTS_TABLE, PAPERS_TABLE):
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table.name} "
            f"(corpusid INTEGER PRIMARY KEY, {table.column} TEXT)"
        )

    return conn


def load_abstracts(
    conn: sqlite3.Connection, abstracts_dir: Path, max_abstracts: int | None
) -> None:
    """Load abstracts into database, processing in batches."""
    processed = 0

    for path in tqdm(list(abstracts_dir.glob("*.gz")), desc="Abstracts", position=0):
        batch: list[tuple[int, str]] = []

        with gzip.open(path, "rt") as f:
            n_lines = count_lines(f)

            for line in tqdm(f, total=n_lines, position=1, leave=False):
                if max_abstracts is not None and processed >= max_abstracts:
                    db_save_batch(conn, ABSTRACTS_TABLE, batch)
                    return

                entry = json.loads(line)
                batch.append((entry["corpusid"], entry["abstract"]))
                processed += 1

                if len(batch) >= ABSTRACT_BATCH_SIZE:
                    db_save_batch(conn, ABSTRACTS_TABLE, batch)
                    batch = []

        if batch:  # Insert any remaining items
            db_save_batch(conn, ABSTRACTS_TABLE, batch)


def count_lines(f: TextIO) -> int:
    n_lines = sum(1 for _ in f)
    f.seek(0)  # Reset to beginning file pointer after counting
    return n_lines


def db_save_batch(
    conn: sqlite3.Connection, table: Table, batch: Iterable[tuple[int, str]]
) -> None:
    conn.executemany(
        f"INSERT OR REPLACE INTO {table.name} (corpusid, {table.column}) VALUES (?, ?)",
        batch,
    )
    conn.commit()


def load_papers(
    conn: sqlite3.Connection, papers_dir: Path, max_papers: int | None
) -> None:
    """Load papers into database, processing in batches."""
    processed = 0

    for path in tqdm(list(papers_dir.glob("*.gz")), desc="Papers"):
        batch: list[tuple[int, str]] = []

        with gzip.open(path, "rt") as f:
            n_lines = count_lines(f)

            for line in tqdm(f, total=n_lines, position=1, leave=False):
                if max_papers is not None and processed >= max_papers:
                    db_save_batch(conn, PAPERS_TABLE, batch)
                    return

                entry = json.loads(line)
                batch.append((entry["corpusid"], line))
                processed += 1

                if len(batch) >= PAPER_BATCH_SIZE:
                    db_save_batch(conn, PAPERS_TABLE, batch)
                    batch = []

        if batch:  # Insert any remaining items
            db_save_batch(conn, PAPERS_TABLE, batch)


def process_matches(conn: sqlite3.Connection, output_dir: Path) -> None:
    """Process matches in batches and save to files."""
    cursor = conn.execute("""
        SELECT p.payload, a.abstract
        FROM papers as p
        LEFT JOIN abstracts as a
            ON p.corpusid = a.corpusid
    """)

    batch: list[dict[str, Any]] = []
    batch_num = 0

    for payload, abstract in tqdm(cursor, desc="Processing matches"):
        paper = json.loads(payload)
        batch.append(paper | {"abstract": abstract})

        if len(batch) >= OUTPUT_BATCH_SIZE:
            file_save_batch(batch, output_dir, batch_num)
            batch_num += 1
            batch = []

    if batch:  # Save any remaining items
        file_save_batch(batch, output_dir, batch_num)


def file_save_batch(
    batch: list[dict[str, Any]], output_dir: Path, batch_num: int
) -> None:
    """Save a batch of matched papers to a compressed file."""
    output_path = output_dir / f"matched_papers_{batch_num:05d}.json.gz"
    with gzip.open(output_path, "wt") as file:
        json.dump(batch, file)


if __name__ == "__main__":
    app()
