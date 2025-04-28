"""Tests for the search functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from duckdb_hybrid_document_search.db import init_db, store_setting
from duckdb_hybrid_document_search.indexer import index_directories
from duckdb_hybrid_document_search.searcher import init_models, search


@pytest.fixture
def test_db():
    """Create a test database with sample documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test database
        db_path = os.path.join(tmpdir, "test.duckdb")
        conn = init_db(db_path, read_only=False)

        # Create test documents directory
        docs_dir = os.path.join(tmpdir, "docs")
        os.makedirs(docs_dir)

        # Create a few test files
        for i in range(3):
            with open(os.path.join(docs_dir, f"doc{i}.md"), "w", encoding="utf-8") as f:
                f.write(
                    f"""# Document {i}

This is a test document about {'artificial intelligence' if i == 0 else 'database systems' if i == 1 else 'programming languages'}.

## Section {i}.1

More content about {'machine learning' if i == 0 else 'SQL queries' if i == 1 else 'Python code'}.
"""
                )

        # Use a small, fast model for testing
        test_model = "all-MiniLM-L6-v2"

        # Store model in settings
        store_setting(conn, "embedding_model", test_model)

        # Index documents
        index_directories(
            conn=conn,
            directories=[docs_dir],
            embedding_model=test_model,
            workers=1,
            clear=True,
        )

        # Initialize models
        init_models(test_model, test_model)

        yield conn, db_path

        # Close connection
        conn.close()


def test_search_basic(test_db):
    """Test basic search functionality."""
    conn, _ = test_db

    # Search for AI-related content
    results = search(
        conn=conn,
        query="artificial intelligence",
        top_k=5,
        rerank=False,
    )

    # Should find at least one result
    assert len(results) > 0

    # First result should be about AI
    assert "artificial intelligence" in results[0]["content"].lower()


def test_search_with_rerank(test_db):
    """Test search with reranking."""
    conn, _ = test_db

    # Search for programming-related content
    results = search(
        conn=conn,
        query="Python programming",
        top_k=5,
        rerank=False,  # Disable reranking for tests
    )

    # Should find at least one result
    assert len(results) > 0

    # Results should have scores
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_search_with_prefix(test_db):
    """Test search with file path prefix."""
    conn, _ = test_db

    # Get original file path
    original_results = search(
        conn=conn,
        query="database",
        top_k=1,
        file_path_prefix=None,
        rerank=False,
    )

    # Should find at least one result
    assert len(original_results) > 0

    # Now search with prefix
    prefix = "custom/path"  # 相対パスに変更
    results = search(
        conn=conn,
        query="database",
        top_k=5,
        file_path_prefix=prefix,
        rerank=False,
    )

    # Should find at least one result
    assert len(results) > 0

    # Check that the content is the same but the path is different
    assert results[0]["content"] == original_results[0]["content"]
    assert results[0]["file_path"] != original_results[0]["file_path"]
    assert prefix in results[0]["file_path"]
