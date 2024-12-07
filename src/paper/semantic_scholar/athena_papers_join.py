"""Create Athena tables for S2ORC datasets: `papers`, `abstracts` and `papers_enriched`.

Tables created:
- `papers`: gzipped NDJSON files from the `papers` S2ORC dataset.
- `abstracts`: gzipped NDJSON files from the `abstracts` S2ORC dataset.
- `papers_enriched`: `papers` joined with `abstracts`. The output is all columns in
  `papers` with the `abstract` column.

In all cases, the existing tables are deleted before creating anew. In the case of
`papers_enriched`, the underlying files are deleted, too.

Requires the following environment variables:
- ATHENA_DATABASE: name of the Athena database where tables will be created.
- ATHENA_OUTPUT_BUCKET: name of the S3 bucket where query results will be stored. Note:
  this is the _name_, not the URL.
- AWS_REGION: name of the region with the S3 buckets.
"""

from __future__ import annotations

import asyncio
import contextlib
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Self

import aioboto3
import typer
from dotenv import load_dotenv
from tqdm import tqdm

from paper.util import mustenv
from paper.util.cli import die

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)


@app.command(help=__doc__)
def main(
    confirm: Annotated[
        bool,
        typer.Option(
            "--assume-yes", "-y", help="Automatically answer yes to all prompts."
        ),
    ] = False,
) -> None:
    """Create Athena tables for S2ORC datasets."""
    asyncio.run(_main(confirm))


async def _main(confirm: bool) -> None:
    load_dotenv(interpolate=True)

    env = mustenv("ATHENA_DATABASE", "ATHENA_OUTPUT_BUCKET", "AWS_REGION")
    database = env["ATHENA_DATABASE"]
    output_bucket = env["ATHENA_OUTPUT_BUCKET"]
    aws_region = env["AWS_REGION"]

    athena = AthenaWrapper(
        database=database,
        output_bucket=output_bucket,
        region=aws_region,
        auto_confirm=confirm,
    )

    echo(f"Database ensurance started: '{database}'")
    await ensure_database(athena)
    echo(f"Database ensurance done: '{database}'\n")

    abstracts_config = TableConfig(
        name="abstracts",
        s3_location="s3://s2orc-datasets/abstracts/",
        columns={
            "corpusid": "bigint",
            "abstract": "string",
        },
    )
    echo("Table creation started: 'abstracts'")
    await create_table(athena, abstracts_config)
    echo("Table creation done: 'abstracts'\n")

    papers_config = TableConfig(
        name="papers",
        s3_location="s3://s2orc-datasets/papers/",
        columns={
            "corpusid": "bigint",
            "title": "string",
            "authors": "array<struct<authorId:string,name:string>>",
            "venue": "string",
            "year": "int",
            "referencecount": "int",
            "citationcount": "int",
            "influentialcitationcount": "int",
        },
    )
    echo("Table creation start: 'papers'")
    await create_table(athena, papers_config)
    echo("Table creation done: 'papers'\n")

    joined_name = "paper_enriched"
    echo(f"Table creation start: '{joined_name}'")
    await create_joined_table(athena, joined_name)
    echo(f"Table creation done: '{joined_name}'\n")


class AthenaWrapper:
    """Wrap Athena and S3 sessions for common operations."""

    def __init__(
        self,
        *,
        database: str,
        output_bucket: str,
        region: str,
        auto_confirm: bool = False,
    ) -> None:
        self._database = database
        self._output_bucket = output_bucket
        self._auto_confirm = auto_confirm
        self._session = aioboto3.Session()
        self._region = region

    async def __aenter__(self) -> Self:
        """Open Athena and S3 sessions.

        Ensures the S3 bucket exists, creating it if not.
        """
        self._athena = await self._session.client(  # type: ignore
            "athena", region_name=self._region
        ).__aenter__()
        self._s3 = await self._session.client(  # type: ignore
            "s3", region_name=self._region
        ).__aenter__()
        await self._ensure_bucket(self._region)
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> bool | None:
        """Close Athena and S3 sessions."""
        await self._athena.__aexit__(exc_type, exc_val, exc_tb)
        await self._s3.__aexit__(exc_type, exc_val, exc_tb)

    async def clean_location(self, prefix: str) -> None:
        """Delete all objects with a given `prefix`."""
        paginator = self._s3.get_paginator("list_objects_v2")
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"

        total_objects = 0
        async for page in paginator.paginate(Bucket=self._output_bucket, Prefix=prefix):
            if "Contents" in page:
                total_objects += len(page["Contents"])

        if total_objects == 0:
            echo(f"No objects found with prefix '{prefix}'")
            return

        if not confirm_action(
            f"This will delete {total_objects} objects with prefix '{prefix}'. Proceed?",
            auto_confirm=self._auto_confirm,
        ):
            die("Aborting: we can't run a query with existing data.")

        async for page in paginator.paginate(Bucket=self._output_bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            objects = [
                {"Key": key} for obj in page["Contents"] if (key := obj.get("Key"))
            ]
            if objects:
                echo(f"Deleting {len(objects)} objects with prefix '{prefix}'...")
                await self._s3.delete_objects(
                    Bucket=self._output_bucket,
                    Delete={"Objects": objects, "Quiet": True},  # type: ignore
                )

        echo(f"Cleanup complete for prefix '{prefix}'")

    async def execute_query(self, query: str, *, show_progress: bool = False) -> None:
        """Execute an SQL query and save to the specified output bucket."""
        output_bucket_url = f"s3://{self._output_bucket}"
        query = query.format(database=self._database, output_bucket=output_bucket_url)

        response = await self._athena.start_query_execution(
            QueryString=textwrap.dedent(query),
            ResultConfiguration={"OutputLocation": output_bucket_url},
            QueryExecutionContext={"Database": self._database},
        )

        query_execution_id = response["QueryExecutionId"]
        last_bytes = 0
        pbar = None

        while True:
            response = await self._athena.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            execution = response.get("QueryExecution", {})
            state = execution.get("Status", {}).get("State")

            if show_progress:
                stats = execution.get("Statistics", {})
                data_scanned = stats.get("DataScannedInBytes", 0)
                if data_scanned > last_bytes:
                    mb_scanned = data_scanned / (1024 * 1024)
                    elapsed = stats.get("EngineExecutionTimeInMillis", 0) / 1000

                    if pbar is None:
                        pbar = tqdm(
                            desc="Scanning",
                            unit="MB",
                            unit_scale=True,
                            unit_divisor=1024,
                        )

                    pbar.update(mb_scanned - (last_bytes / (1024 * 1024)))
                    last_bytes = data_scanned

            if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                if pbar is not None:
                    pbar.close()

                if state != "SUCCEEDED":
                    status = execution.get("Status", {})
                    raise RuntimeError(
                        f"Query failed with state {state}: "
                        f"{status.get('StateChangeReason', 'No reason provided')}"
                    )

                if show_progress:
                    stats = execution.get("Statistics", {})
                    data_scanned = stats.get("DataScannedInBytes", 0)
                    mb_scanned = data_scanned / (1024 * 1024)
                    elapsed = stats.get("EngineExecutionTimeInMillis", 0) / 1000
                    echo(
                        f"Final stats: {mb_scanned:.1f} MB scanned in {elapsed:.1f}s "
                        f"({mb_scanned/elapsed:.1f} MB/s)"
                    )
                break

            await asyncio.sleep(5)

    async def _ensure_bucket(self, region: str) -> None:
        with contextlib.suppress(
            self._s3.exceptions.BucketAlreadyExists,
            self._s3.exceptions.BucketAlreadyOwnedByYou,
        ):
            await self._s3.create_bucket(
                Bucket=self._output_bucket,
                CreateBucketConfiguration={"LocationConstraint": region},  # type: ignore
            )


def confirm_action(message: str, *, auto_confirm: bool) -> bool:
    """If auto-confirm is disabled, ask for user confirmation. Otherwise, return True.

    Args:
        message: The confirmation message to show.
        auto_confirm: If true, skips asking the user for confirmation and just do it.

    Returns:
        True if confirmed, False otherwise.
    """
    if auto_confirm:
        return True

    response = typer.prompt(f"{message} [y/N]", default="n", show_default=False)
    return response.lower().strip() == "y"


async def ensure_database(athena: AthenaWrapper) -> None:
    """Create the Athena database if it doesn't exist."""
    query = "CREATE DATABASE IF NOT EXISTS {database}"
    await athena.execute_query(query)


@dataclass(frozen=True, kw_only=True)
class TableConfig:
    """Configuration for creating an Athena table."""

    # Name of the created table.
    name: str
    # S3 location with the source data.
    s3_location: str
    # Columns for the resulting table. Doesn't have to be all of the columns in the data.
    columns: dict[str, str]


async def create_table(athena: AthenaWrapper, config: TableConfig) -> None:
    """Create an Athena table from the configuration. If it exists, it's dropped first."""
    await drop_table(athena, config.name)

    columns_sql = ",\n    ".join(
        f"`{name}` {type_}" for name, type_ in config.columns.items()
    )

    query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {{database}}.{config.name} (
        {columns_sql}
    )
    ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
    LOCATION '{config.s3_location}'
    TBLPROPERTIES ('has_encrypted_data'='false')
    """

    await athena.execute_query(query)


async def drop_table(athena: AthenaWrapper, name: str) -> None:
    """Drop Athena table `name` if it exists."""
    drop_query = f"DROP TABLE IF EXISTS {{database}}.{name}"
    echo(f"Dropping table if exists: '{name}'")
    await athena.execute_query(drop_query)


async def create_joined_table(athena: AthenaWrapper, name: str) -> None:
    """Create the final joined table by joining papers and abstracts.

    If it already exists, it's dropped and its associated files are deleted.
    """
    await drop_table(athena, name)
    await athena.clean_location(name)

    query = f"""
    CREATE TABLE {{database}}.{name}
    WITH (
        format = 'PARQUET',
        parquet_compression = 'SNAPPY',
        external_location = '{{output_bucket}}/{name}/'
    ) AS
    SELECT
        p.corpusid,
        p.title,
        p.authors,
        p.venue,
        p.year,
        p.referencecount,
        p.citationcount,
        p.influentialcitationcount,
        a.abstract
    FROM {{database}}.papers p
    INNER JOIN {{database}}.abstracts a
    ON p.corpusid = a.corpusid
    """

    echo(f"Starting join operation for '{name}' (this may take a while)...")
    await athena.execute_query(query, show_progress=True)


def echo(message: Any) -> None:
    """Print `message` to stderr with a timestamp."""
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    typer.echo(f"[{ts}] {message}", err=True)


if __name__ == "__main__":
    app()
