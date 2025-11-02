import argparse
import logging
import os
import sqlite3
from contextlib import closing
from typing import Iterable, List, Sequence, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load COD data from SQLite dump into PostgreSQL.")
    parser.add_argument(
        "--sqlite-path",
        type=str,
        required=True,
        help="Path to the COD SQLite dump (e.g., db_raw/cod2205.sq).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of rows to transfer per batch.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate destination table before loading.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="cod_entries",
        help="Destination table name inside PostgreSQL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def get_pg_connection() -> psycopg2.extensions.connection:
    host = os.getenv("DATABASE_HOST", "localhost")
    port = int(os.getenv("DATABASE_PORT", "5432"))
    name = os.getenv("DATABASE_NAME", "phaseid")
    user = os.getenv("DATABASE_USER", "phaseid")
    password = os.getenv("DATABASE_PASSWORD", "phaseid")
    logger.debug("Connecting to PostgreSQL at %s:%s/%s as %s", host, port, name, user)
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=name,
        user=user,
        password=password,
    )


def ensure_destination(cursor: psycopg2.extensions.cursor, table_name: str) -> None:
    create_stmt = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {table} (
            entry_id INTEGER PRIMARY KEY,
            name TEXT,
            mineral_name TEXT,
            chemical_formula TEXT,
            space_group TEXT,
            quality TEXT
        );
        """
    ).format(table=sql.Identifier(table_name))
    cursor.execute(create_stmt)

    index_stmt = sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {index_name} ON {table} (mineral_name);
        """
    ).format(
        table=sql.Identifier(table_name),
        index_name=sql.Identifier(f"{table_name}_mineral_name_idx"),
    )
    cursor.execute(index_stmt)


def truncate_destination(cursor: psycopg2.extensions.cursor, table_name: str) -> None:
    cursor.execute(
        sql.SQL("TRUNCATE TABLE {table};").format(table=sql.Identifier(table_name))
    )


def iter_sqlite_rows(
    connection: sqlite3.Connection, chunk_size: int
) -> Iterable[List[Tuple]]:
    query = """
        SELECT id, name, mineralname, chemical_formula, spacegroup, quality
        FROM id
        ORDER BY id;
    """
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(query)

    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        yield [(row["id"], row["name"], row["mineralname"], row["chemical_formula"], row["spacegroup"], row["quality"]) for row in rows]


def load_rows(
    cursor: psycopg2.extensions.cursor,
    table_name: str,
    rows: Sequence[Tuple],
) -> None:
    insert_stmt = sql.SQL(
        """
        INSERT INTO {table} (entry_id, name, mineral_name, chemical_formula, space_group, quality)
        VALUES %s
        ON CONFLICT (entry_id) DO UPDATE SET
            name = EXCLUDED.name,
            mineral_name = EXCLUDED.mineral_name,
            chemical_formula = EXCLUDED.chemical_formula,
            space_group = EXCLUDED.space_group,
            quality = EXCLUDED.quality;
        """
    ).format(table=sql.Identifier(table_name))
    execute_values(cursor, insert_stmt.as_string(cursor), rows, page_size=len(rows))


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    sqlite_path = os.path.abspath(args.sqlite_path)
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite dump not found at {sqlite_path}")

    logger.info("Loading COD data from %s", sqlite_path)
    with closing(sqlite3.connect(sqlite_path)) as sqlite_conn, closing(get_pg_connection()) as pg_conn:
        pg_conn.autocommit = False
        with closing(pg_conn.cursor()) as pg_cursor:
            ensure_destination(pg_cursor, args.table_name)
            if args.truncate:
                logger.info("Truncating destination table %s", args.table_name)
                truncate_destination(pg_cursor, args.table_name)

            total_rows = 0
            try:
                for batch in iter_sqlite_rows(sqlite_conn, args.chunk_size):
                    if not batch:
                        continue
                    load_rows(pg_cursor, args.table_name, batch)
                    total_rows += len(batch)
                    logger.debug("Loaded %d rows (running total: %d)", len(batch), total_rows)
                pg_conn.commit()
            except Exception:
                pg_conn.rollback()
                logger.exception("Failed to load COD data; transaction rolled back.")
                raise

    logger.info("Completed COD load: %d rows upserted into %s.", total_rows, args.table_name)


if __name__ == "__main__":
    main()
