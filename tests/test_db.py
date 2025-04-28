"""Tests for the database functionality."""

import os
import tempfile

import pytest

from duckdb_hybrid_document_search.db import (
    clear_documents,
    get_setting,
    init_db,
    store_setting,
)


def test_init_db():
    """Test database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")

        # Initialize database
        conn = init_db(db_path, read_only=False)

        # Check if tables exist
        result = conn.execute(
            """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN ('documents', 'settings')
        """
        ).fetchall()

        # Should have both tables
        assert len(result) == 2

        # Close connection
        conn.close()


def test_store_and_get_setting():
    """Test storing and retrieving settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")

        # Initialize database
        conn = init_db(db_path, read_only=False)

        # Store a setting
        key = "test_key"
        value = "test_value"
        store_setting(conn, key, value)

        # Retrieve the setting
        retrieved = get_setting(conn, key)

        # Should match
        assert retrieved == value

        # Retrieve non-existent setting
        non_existent = get_setting(conn, "non_existent")

        # Should be None
        assert non_existent is None

        # Close connection
        conn.close()


def test_clear_documents():
    """Test clearing documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")

        # Initialize database
        conn = init_db(db_path, read_only=False)

        # Insert a test document
        from ulid import ULID

        doc_id = str(ULID())
        conn.execute(
            """
        INSERT INTO documents (doc_id, file_path, content, tokens)
        VALUES (?, ?, ?, ?)
        """,
            [doc_id, "test.md", "Test content", "test content"],
        )

        # Check if document exists
        count_before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count_before > 0

        # Clear documents
        clear_documents(conn)

        # Check if documents are cleared
        count_after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count_after == 0

        # Close connection
        conn.close()
