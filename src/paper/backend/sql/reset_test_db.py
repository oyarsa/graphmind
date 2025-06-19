"""Reset the test database by dropping tables, creating schema, and seeding data."""

import asyncio
import tempfile
from importlib import resources
from pathlib import Path

from paper.backend.db import DEFAULT_DB_PARAMS, execute_query, seed
from paper.backend.generate_data import main as generate_data


def execute(sql: str) -> None:
    """Execute SQL on default database credentials."""
    asyncio.run(
        execute_query(
            dbname=DEFAULT_DB_PARAMS["dbname"],
            user=DEFAULT_DB_PARAMS["user"],
            password=DEFAULT_DB_PARAMS["password"],
            host=DEFAULT_DB_PARAMS["host"],
            port=DEFAULT_DB_PARAMS["port"],
            sql=sql,
        )
    )


ROOT = Path(__file__).parent.parent


def reset_test_db() -> None:
    """Reset the test database."""

    print("Dropping tables")
    try:
        execute("DROP TABLE related; DROP TABLE paper;")
    except Exception as e:
        print(f"Tables might not exist yet: {e}")

    print("Creating schema")
    schema = (
        resources.files("paper.backend.sql").joinpath("basic_schema.sql").read_text()
    )
    execute(schema)

    print("Generating test data")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        data_path = tmp_path / "database.json"
        generate_data(output_file=data_path)

        print("Seeding database")
        seed(seed_file=data_path)

    print("Database reset complete!")


if __name__ == "__main__":
    reset_test_db()
