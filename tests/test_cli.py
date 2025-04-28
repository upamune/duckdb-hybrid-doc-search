"""Tests for the CLI functionality."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from duckdb_hybrid_document_search.cli import app
from duckdb_hybrid_document_search.db import init_db, store_setting


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


def test_version(runner):
    """Test version command."""
    with patch("duckdb_hybrid_document_search.cli.__version__", "0.1.0"):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        # Rich形式の出力を考慮
        assert "0.1" in result.stdout and "0" in result.stdout


def test_index_command_validation(runner):
    """Test index command validation."""
    # Test with non-existent directory
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "non_existent")
        result = runner.invoke(app, ["index", non_existent, "--embedding-model", "test-model"])
        assert result.exit_code != 0

        # Create a test file (not a directory)
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Test with file instead of directory
        result = runner.invoke(app, ["index", test_file, "--embedding-model", "test-model"])
        assert result.exit_code != 0


def test_serve_command_validation(runner):
    """Test serve command validation."""
    # Test with non-existent database
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_db = os.path.join(tmpdir, "non_existent.duckdb")
        result = runner.invoke(app, ["serve", "--db", non_existent_db])
        assert result.exit_code != 0


def test_search_command(runner):
    """Test search command with direct query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")

        # Create a mock database
        with (
            patch("duckdb_hybrid_document_search.cli.init_db") as mock_init_db,
            patch("duckdb_hybrid_document_search.cli.perform_search") as mock_perform_search,
            patch("duckdb_hybrid_document_search.cli.init_models") as mock_init_models,
            patch("os.path.exists", return_value=True),
        ):

            # Mock database connection and query result
            mock_conn = MagicMock()
            mock_init_db.return_value = mock_conn
            mock_conn.execute.return_value.fetchone.return_value = ["test-model"]

            # Test search command with direct query
            result = runner.invoke(app, ["search", "--db", db_path, "--query", "test query"])

            assert result.exit_code == 0

            # Verify function calls
            mock_init_models.assert_called_once()
            mock_perform_search.assert_called_once()

            # Check that perform_search was called with the right arguments
            args, kwargs = mock_perform_search.call_args
            assert kwargs["query"] == "test query"
            assert kwargs["top_k"] == 5  # Default value
            assert kwargs["rerank"] == True  # Default is True (not no_rerank)


# インタラクティブモードのテストは終了しないため削除


def test_search_command_validation(runner):
    """Test search command validation."""
    # Test with non-existent database
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_db = os.path.join(tmpdir, "non_existent.duckdb")
        result = runner.invoke(app, ["search", "--db", non_existent_db, "--query", "test query"])
        assert result.exit_code != 0


def test_perform_search():
    """Test the perform_search function."""
    from duckdb_hybrid_document_search.cli import perform_search

    # Mock search results
    mock_results = [
        {
            "doc_id": "1",
            "file_path": "test.md",
            "header_path": "Test/Section",
            "line_start": 1,
            "line_end": 10,
            "content": "Test content",
            "score": 0.95,
        }
    ]

    # Mock console and search function
    console = MagicMock()
    conn = MagicMock()

    with patch("duckdb_hybrid_document_search.searcher.search", return_value=mock_results):
        # Call the function
        perform_search(
            conn=conn,
            query="test query",
            top_k=5,
            file_path_prefix="/test",
            rerank=True,
            embedding_model="test-model",
            rerank_model="test-reranker",
            console=console,
        )

        # Verify console output calls
        console.print.assert_any_call("[bold]Search results for: [cyan]test query[/cyan][/bold]")
        console.print.assert_any_call("[bold]1. [green]test.md[/green][/bold]")
        console.print.assert_any_call("   Section: Test/Section")
        console.print.assert_any_call("   Lines: 1-10")
        console.print.assert_any_call("   Score: 0.9500")

        # Test with no results
        with patch("duckdb_hybrid_document_search.searcher.search", return_value=[]):
            perform_search(
                conn=conn,
                query="no results",
                top_k=5,
                file_path_prefix="/test",
                rerank=True,
                embedding_model="test-model",
                rerank_model="test-reranker",
                console=console,
            )

            console.print.assert_any_call("[yellow]No results found[/yellow]")
