"""DuckDB connection and schema DDL."""

import logging
from typing import Optional

import duckdb
from ulid import ULID

from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)


def init_db(db_path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection and create schema if needed.

    Args:
        db_path: Path to DuckDB database file
        read_only: Whether to open in read-only mode

    Returns:
        DuckDB connection
    """
    logger.info(f"Connecting to DuckDB at {db_path} (read_only={read_only})")
    conn = duckdb.connect(db_path, read_only=read_only)

    # Create schema if not in read-only mode
    if not read_only:
        create_schema(conn)

    return conn


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the database schema.

    Args:
        conn: DuckDB connection
    """
    logger.info("Creating database schema")

    # Create documents table
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS documents (
      doc_id      TEXT PRIMARY KEY,
      file_path   TEXT NOT NULL,
      header_path TEXT,
      line_start  INTEGER,
      line_end    INTEGER,
      content     TEXT,
      tokens      TEXT,
      embedding   FLOAT[384]
    );
    """
    )

    # Create settings table
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS settings (
      key   TEXT PRIMARY KEY,
      value TEXT
    );
    """
    )

    # Install and load FTS extension
    conn.execute("INSTALL fts;")
    conn.execute("LOAD fts;")

    # Create FTS index
    conn.execute(
        """
    PRAGMA create_fts_index(
      'documents', 'doc_id', 'tokens',
      stemmer='none', stopwords='none'
    );
    """
    )

    # Install and load VSS extension
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    # Enable HNSW persistence for VSS index
    conn.execute("SET hnsw_enable_experimental_persistence=true;")

    # Create VSS index
    conn.execute(
        """
    CREATE INDEX IF NOT EXISTS idx_embedding
      ON documents USING HNSW (embedding);
    """
    )

    logger.info("Schema creation complete")


def store_setting(conn: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    """Store a setting in the settings table.

    Args:
        conn: DuckDB connection
        key: Setting key
        value: Setting value
    """
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", [key, value])


def get_setting(conn: duckdb.DuckDBPyConnection, key: str) -> Optional[str]:
    """Get a setting from the settings table.

    Args:
        conn: DuckDB connection
        key: Setting key

    Returns:
        Setting value or None if not found
    """
    result = conn.execute("SELECT value FROM settings WHERE key = ?", [key]).fetchone()

    if result:
        return result[0]

    return None


def clear_documents(conn: duckdb.DuckDBPyConnection) -> None:
    """Clear all documents from the database.

    Args:
        conn: DuckDB connection
    """
    logger.info("Clearing all documents from database")
    conn.execute("DELETE FROM documents")
