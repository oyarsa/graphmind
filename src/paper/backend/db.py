"""Manage database operations."""

import asyncio
import itertools
import math
import sys
from collections.abc import Collection, Mapping
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Annotated, LiteralString, cast, override

import typer
from psycopg.errors import UniqueViolation
from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool
from tqdm import tqdm

from paper.backend.generate_data import StaticDatabase
from paper.backend.model import (
    Paper,
    PaperId,
    PaperNeighbour,
    PaperSearchResult,
    Related,
    RelatedType,
)

DEFAULT_DB_PARAMS: Mapping[str, str] = {
    "dbname": "explorer",
    "user": "postgres",
    "password": "dev",
    "host": "localhost",
    "port": "5432",
}


class DatabaseManager(AbstractAsyncContextManager["DatabaseManager"]):
    """Repository for papers using PostgreSQL."""

    _PAPER_INSERT_SQL = """
        INSERT INTO paper (
            id, title, year, authors, abstract, venue, citation_count, doi, pdf_url
        )
        VALUES (
            %(id)s, %(title)s, %(year)s, %(authors)s, %(abstract)s, %(venue)s,
            %(citation_count)s, %(doi)s, %(pdf_url)s
        )
    """

    _RELATED_INSERT_SQL = """
        INSERT INTO related (source, target, type, similarity)
        VALUES (%(source)s, %(target)s, %(type)s, %(similarity)s)
    """

    def __init__(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str,
        port: str,
        max_size: int = 10,
    ) -> None:
        """Create new database manager.

        Args:
            dbname: Name of the PostgreSQL database.
            user: Database username.
            password: Database password.
            host: Database host address.
            port: Database port number.
            max_size: Maximum number of connections in the pool.

        Note:
            Must either use as context manager or call .open() explicitly.
        """
        connstr = f"{dbname=} {user=} {password=} {host=} {port=}"
        self.pool = AsyncConnectionPool(
            connstr, min_size=1, max_size=max_size, open=False
        )

    async def open(self) -> None:
        """Open the database connection pool."""
        await self.pool.open()

    async def execute(self, sql: LiteralString) -> None:
        """Execute DDL query on the database.

        Args:
            sql: SQL query to execute.
        """
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql)
            await conn.commit()

    async def search_papers(
        self, query_text: str, limit: int
    ) -> list[PaperSearchResult]:
        """Search papers using full-text search on title and abstract.

        Uses PostgreSQL's text search capabilities with weighted ranking.
        Results are ordered by relevance score and citation count.

        Args:
            query_text: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of paper search results with relevance scores.
        """

        search_sql = """
        SELECT
            id, title, year, authors,
            ts_rank(search_vector, plainto_tsquery('english', %(query)s))
                AS relevance
        FROM paper
        WHERE search_vector @@ plainto_tsquery('english', %(query)s)
        ORDER BY relevance DESC, citation_count DESC
        LIMIT %(limit)s
        """

        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=class_row(PaperSearchResult)) as cur,
        ):
            await cur.execute(search_sql, {"query": query_text, "limit": limit})
            return await cur.fetchall()

    async def get_related(
        self, paper_id: PaperId, limit: int, type_: RelatedType | None = None
    ) -> list[Related]:
        """Get all related papers from paper_id (both source and target).

        Args:
            paper_id: ID of the paper to find relations for.
            limit: Maximum number of relations to return.
            type_: Optional relation type filter (citation or semantic).

        Returns:
            List of related paper relations ordered by similarity score.
        """
        sql = """
        SELECT source, target, type, similarity
        FROM related
        WHERE (source = %(paper_id)s OR target = %(paper_id)s)
          AND type = COALESCE(%(type)s, type)
        ORDER BY similarity desc
        LIMIT %(limit)s
        """

        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=class_row(Related)) as cur,
        ):
            await cur.execute(
                sql,
                {
                    "paper_id": paper_id,
                    "type": type_.value if type_ else None,
                    "limit": limit,
                },
            )
            return await cur.fetchall()

    async def get_neighbours(
        self, paper_id: PaperId, limit: int, type_: RelatedType
    ) -> list[PaperNeighbour]:
        """Get related papers with full paper information.

        For citations: returns papers that cite the given paper (paper_id is source).
        For semantic: returns papers related in either direction (symmetric).

        Args:
            paper_id: ID of the paper to find neighbours for.
            limit: Maximum number of neighbours to return.
            type_: Type of relation (citation or semantic).

        Returns:
            List of neighbouring papers with full information and similarity scores.
        """
        if type_ is RelatedType.CITATION:
            # For citations, only get papers that cite this paper
            sql = """
            SELECT
                p.id, p.title, p.year, p.authors, p.abstract, p.venue, p.citation_count,
                p.doi, p.pdf_url, r.similarity, r.type
            FROM related as r
            JOIN paper as p ON p.id = r.target
            WHERE r.source = %(paper_id)s
              AND r.type = %(type)s
            ORDER BY r.similarity desc
            LIMIT %(limit)s
            """
        elif type_ is RelatedType.SEMANTIC:
            # For semantic relationships, get papers in either direction
            sql = """
            SELECT
                p.id, p.title, p.year, p.authors, p.abstract, p.venue, p.citation_count,
                p.doi, p.pdf_url, r.similarity, r.type
            FROM related as r
            JOIN paper as p ON (
                (r.source = %(paper_id)s AND p.id = r.target) OR
                (r.target = %(paper_id)s AND p.id = r.source)
            )
            WHERE r.type = %(type)s
            ORDER BY r.similarity desc
            LIMIT %(limit)s
            """

        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=class_row(PaperNeighbour)) as cur,
        ):
            await cur.execute(
                sql,
                {
                    "paper_id": paper_id,
                    "type": type_.value,
                    "limit": limit,
                },
            )
            return await cur.fetchall()

    async def get_paper(self, paper_id: PaperId) -> Paper | None:
        """Get full information for a paper by ID.

        Args:
            paper_id: ID of the paper to retrieve.

        Returns:
            Paper object with full information, or None if not found.
        """
        sql = """
        SELECT
            id, title, year, authors, abstract, venue, citation_count, doi, pdf_url
        FROM paper
        WHERE id = %(paper_id)s
        """

        async with (
            self.pool.connection() as conn,
            conn.cursor(row_factory=class_row(Paper)) as cur,
        ):
            await cur.execute(sql, {"paper_id": paper_id})
            return await cur.fetchone()

    async def insert_paper(self, paper: Paper) -> None:
        """Insert a single paper into the database.

        Args:
            paper: Paper object to insert.
        """
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(self._PAPER_INSERT_SQL, paper.model_dump())
            await conn.commit()

    async def delete_paper(self, paper_id: str) -> None:
        """Delete a paper by ID.

        Args:
            paper_id: ID of the paper to delete.
        """

        delete_sql = """
        DELETE FROM paper
        WHERE id = %(paper_id)s
        """

        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(delete_sql, {"paper_id": paper_id})
            await conn.commit()

    async def insert_related(self, related: Related) -> None:
        """Insert a single paper relation.

        Args:
            related: Related paper relation to insert.
        """
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(self._RELATED_INSERT_SQL, related.model_dump())
            await conn.commit()

    async def delete_related(self, source: str, target: str, type_: str) -> None:
        """Delete a paper relation by source, target and type.

        Args:
            source: Source paper ID.
            target: Target paper ID.
            type_: Relation type (citation or semantic).
        """

        delete_sql = """
        DELETE FROM related
        WHERE source = %(source)s
          AND target = %(target)s
          AND type = %(type)s
        """

        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                delete_sql, {"source": source, "target": target, "type": type_}
            )
            await conn.commit()

    async def batch_insert_papers(
        self,
        papers: Collection[Paper],
        chunk_size: int = 1000,
        *,
        progress: bool = False,
    ) -> None:
        """Batch insert multiple papers in chunks with conflict handling.

        Uses ON CONFLICT DO NOTHING to handle duplicate IDs gracefully.

        Args:
            papers: Collection of paper objects to insert.
            chunk_size: Number of papers to insert per batch.
            progress: Whether to show progress bar.
        """
        if not papers:
            return

        insert_sql = self._PAPER_INSERT_SQL + " ON CONFLICT (id) DO NOTHING"

        chunks = itertools.batched(papers, chunk_size)
        if progress:
            chunks = tqdm(
                chunks,
                desc="Inserting papers",
                total=math.ceil(len(papers) / chunk_size),
            )

        async with self.pool.connection() as conn:
            for chunk in chunks:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        insert_sql, [paper.model_dump() for paper in chunk]
                    )
                await conn.commit()

    async def batch_insert_related(
        self,
        relations: Collection[Related],
        chunk_size: int = 5000,
        *,
        progress: bool = False,
    ) -> None:
        """Batch insert multiple paper relations in chunks with conflict handling.

        Uses ON CONFLICT DO NOTHING to handle duplicate relations gracefully.

        Args:
            relations: Collection of related paper relations to insert.
            chunk_size: Number of relations to insert per batch.
            progress: Whether to show progress bar.
        """
        if not relations:
            return

        insert_sql = (
            self._RELATED_INSERT_SQL + " ON CONFLICT (source, target, type) DO NOTHING"
        )

        chunks = itertools.batched(relations, chunk_size)
        if progress:
            chunks = tqdm(
                chunks,
                desc="Inserting relations",
                total=math.ceil(len(relations) / chunk_size),
            )

        async with self.pool.connection() as conn:
            for chunk in chunks:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        insert_sql,
                        [related.model_dump(by_alias=True) for related in chunk],
                    )
                await conn.commit()

    async def close(self) -> None:
        """Close the database connection pool."""
        await self.pool.close()

    @override
    async def __aenter__(self) -> "DatabaseManager":
        await self.open()
        return self

    @override
    async def __aexit__(self, *_: object) -> None:
        await self.close()


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

OPTION_DBNAME = Annotated[
    str,
    typer.Option(
        "--dbname", envvar="XP_DB_NAME", help="Name of the PostgreSQL database."
    ),
]
OPTION_DBUSER = Annotated[
    str,
    typer.Option("--user", "-u", envvar="XP_DB_USER", help="Database username."),
]
# do the same for password, host and port
OPTION_DBPASSWORD = Annotated[
    str,
    typer.Option(
        "--password",
        "-P",
        envvar="XP_DB_PASSWORD",
        help="Database password",
    ),
]
OPTION_DBHOST = Annotated[
    str,
    typer.Option("--host", "-h", envvar="XP_DB_HOST", help="Database host."),
]
OPTION_DBPORT = Annotated[
    str,
    typer.Option("--port", "-p", envvar="XP_DB_PORT", help="Database port."),
]


@app.command(no_args_is_help=True)
def execute(
    file: Annotated[
        Path,
        typer.Argument(help="File with SQL to execute. Use `-` to read from stdin."),
    ],
    dbname: OPTION_DBNAME = DEFAULT_DB_PARAMS["dbname"],
    user: OPTION_DBUSER = DEFAULT_DB_PARAMS["user"],
    password: OPTION_DBPASSWORD = DEFAULT_DB_PARAMS["password"],
    host: OPTION_DBHOST = DEFAULT_DB_PARAMS["host"],
    port: OPTION_DBPORT = DEFAULT_DB_PARAMS["port"],
) -> None:
    """Execute SQL commands from a file or stdin.

    Args:
        file: Path to SQL file or '-' to read from stdin.
        dbname: Database name.
        user: Database username.
        password: Database password.
        host: Database host.
        port: Database port.

    Note:
        The SQL will be executed as-is. Make sure you trust the source.
    """
    if str(file) == "-":
        sql = sys.stdin.read()
    else:
        sql = file.read_text()

    asyncio.run(execute_query(dbname, user, password, host, port, sql))


async def execute_query(
    dbname: str, user: str, password: str, host: str, port: str, sql: str
) -> None:
    """Execute DLL query on database.

    Args:
        dbname: Name of the database on PostgreSQL.
        user: Username for the database.
        password: Password for the database.
        host: Host URL.
        port: Database access port.
        sql: DDL query to execute.

    Returns:
        Nothing. If the query returns a result, it will be ignored.
    """
    async with DatabaseManager(dbname, user, password, host, port) as db:
        print("Executing file...")
        await db.execute(cast(LiteralString, sql))
        print("Done.")


@app.command()
def test(
    dbname: OPTION_DBNAME = DEFAULT_DB_PARAMS["dbname"],
    user: OPTION_DBUSER = DEFAULT_DB_PARAMS["user"],
    password: OPTION_DBPASSWORD = DEFAULT_DB_PARAMS["password"],
    host: OPTION_DBHOST = DEFAULT_DB_PARAMS["host"],
    port: OPTION_DBPORT = DEFAULT_DB_PARAMS["port"],
) -> None:
    """Test database connectivity and operations.

    Inserts test data, performs queries, and cleans up afterwards.

    Args:
        dbname: Database name.
        user: Database username.
        password: Database password.
        host: Database host.
        port: Database port.
    """

    async def _test() -> None:
        async with DatabaseManager(dbname, user, password, host, port) as db:
            paper = Paper(
                id=PaperId("paper123"),
                title="Attention Is All You Need",
                year=2017,
                authors=["Vaswani", "Shazeer", "Parmar"],
                abstract="Introducing transformers...",
                venue="NeurIPS",
                citation_count=50000,
                doi="10.5555/3295222.3295349",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            )
            print("Inserting paper")
            try:
                await db.insert_paper(paper)
            except UniqueViolation as e:
                print(f"PostgreSQL error:\n{e}")
            print("    Done.")

            # Example: Search papers
            print("\nSearching papers...")
            results = await db.search_papers("transformer attention", limit=5)
            for result in results:
                authors_ = ", ".join(result.authors)
                print(
                    f"    [{result.relevance:.3f}] {result.id} | {result.title}"
                    f" ({result.year}) | {authors_}"
                )

            print("\nDeleting paper...")
            await db.delete_paper(paper.id)
            print("    Done.")

    asyncio.run(_test())


@app.command(no_args_is_help=True)
def seed(
    seed_file: Annotated[
        Path, typer.Argument(help="JSON file with seed data to insert.")
    ],
    dbname: OPTION_DBNAME = DEFAULT_DB_PARAMS["dbname"],
    user: OPTION_DBUSER = DEFAULT_DB_PARAMS["user"],
    password: OPTION_DBPASSWORD = DEFAULT_DB_PARAMS["password"],
    host: OPTION_DBHOST = DEFAULT_DB_PARAMS["host"],
    port: OPTION_DBPORT = DEFAULT_DB_PARAMS["port"],
) -> None:
    """Insert seed data from JSON file into the database.

    Loads paper and relation data from a JSON file and inserts them
    into the database using batch operations.

    Args:
        seed_file: Path to JSON file containing seed data.
        dbname: Database name.
        user: Database username.
        password: Database password.
        host: Database host.
        port: Database port.
    """

    async def _seed() -> None:
        async with DatabaseManager(dbname, user, password, host, port) as db:
            seed_data = seed_file.read_text()
            papers = StaticDatabase.model_validate_json(seed_data)

            print(f"Inserting {len(papers.papers)} papers...")
            await db.batch_insert_papers(papers.papers, progress=True)
            print("Papers inserted successfully.")

            print(f"Inserting {len(papers.related)} relations...")
            await db.batch_insert_related(papers.related, progress=True)
            print("Relations inserted successfully.")

    asyncio.run(_seed())


if __name__ == "__main__":
    app()
