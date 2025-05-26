"""Command-line interface for duckdb-hybrid-doc-search."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb
import typer
from rich.console import Console

from duckdb_hybrid_document_search import __version__
from duckdb_hybrid_document_search.db import init_db, store_setting
from duckdb_hybrid_document_search.indexer import index_directories
from duckdb_hybrid_document_search.models.embedding import get_embedding_model
from duckdb_hybrid_document_search.models.reranker import get_reranker_model
from duckdb_hybrid_document_search.searcher import init_models
from duckdb_hybrid_document_search.server import run_server
from duckdb_hybrid_document_search.splitter import SplitterType
from duckdb_hybrid_document_search.utils.logging import setup_logger

app = typer.Typer(
    help="Hybrid FTS + VSS search over Markdown using DuckDB",
    add_completion=False,
)

console = Console()
logger = setup_logger(console=console)


@app.command()
def version():
    """Show version information."""
    console.print(f"duckdb-hybrid-doc-search version [bold]{__version__}[/bold]")


@app.command()
def index(
    directories: List[str] = typer.Argument(
        ...,
        help="Directories containing Markdown files to index",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    db: str = typer.Option(
        "index.duckdb",
        "--db",
        "-d",
        help="Path to DuckDB database file",
    ),
    workers: int = typer.Option(
        os.cpu_count() or 1,
        "--workers",
        "-w",
        help="Number of worker processes for parallel indexing",
        min=1,
        max=os.cpu_count() or 8,
    ),
    embedding_model: str = typer.Option(
        "cl-nagoya/ruri-v3-310m",
        "--embedding-model",
        "-m",
        help="Hugging Face model ID for embeddings",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear existing documents before indexing",
    ),
    splitter: str = typer.Option(
        "llama-index",
        "--splitter",
        "-s",
        help="Splitter type to use (chonkie or llama-index)",
    ),
    path_prefix_to_trim: Optional[str] = typer.Option(
        None,
        "--trim-path-prefix",
        "-t",
        help="Prefix to trim from file paths during indexing (e.g., '/app/')",
    ),
):
    """Index Markdown documents for hybrid search."""
    try:
        # Validate directories
        for directory in directories:
            if not os.path.isdir(directory):
                console.print(f"[red]Error: {directory} is not a directory[/red]")
                sys.exit(1)

        # Initialize database
        conn = init_db(db, read_only=False, embedding_model=embedding_model)

        # Convert splitter string to enum
        splitter_type = SplitterType.LLAMA_INDEX
        if splitter.lower() == "chonkie":
            splitter_type = SplitterType.CHONKIE

        # Index documents
        index_directories(
            conn=conn,
            directories=directories,
            embedding_model=embedding_model,
            workers=workers,
            clear=clear,
            splitter_type=splitter_type,
            path_prefix_to_trim=path_prefix_to_trim,
        )

        console.print(f"[green]Indexing complete. Database saved to {db}[/green]")

    except Exception as e:
        console.print(f"[red]Error during indexing: {str(e)}[/red]")
        sys.exit(1)


@app.command()
def serve(
    db: str = typer.Option(
        "index.duckdb",
        "--db",
        "-d",
        help="Path to DuckDB database file",
    ),
    add_path_prefix: Optional[str] = typer.Option(
        None,
        "--add-path-prefix",
        "-p",
        help="Prefix to add to file paths in search results",
    ),
    remove_path_prefix: Optional[str] = typer.Option(
        None,
        "--remove-path-prefix",
        help="Prefix to remove from file paths in search results",
    ),
    rerank_model: str = typer.Option(
        "cl-nagoya/ruri-v3-reranker-310m",
        "--rerank-model",
        "-r",
        help="Hugging Face model ID for reranking",
    ),
    tool_name: str = typer.Option(
        "search_documents",
        "--tool-name",
        help="Name of the MCP tool",
    ),
    tool_description: str = typer.Option(
        "Search for local documents",
        "--tool-description",
        help="Description of the MCP tool",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport protocol to use (stdio, streamable-http)",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind to for HTTP transport",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        help="Port to bind to for HTTP transport",
    ),
    path: str = typer.Option(
        "/mcp",
        "--path",
        help="Path for streamable-http transport",
    ),
):
    """Start MCP server for document search with specified transport."""
    try:
        if not os.path.exists(db):
            console.print(f"[red]Error: Database file {db} not found[/red]")
            sys.exit(1)

        # Validate transport
        valid_transports = ["stdio", "streamable-http"]
        if transport not in valid_transports:
            console.print(f"[red]Error: Invalid transport '{transport}'. Must be one of: {', '.join(valid_transports)}[/red]")
            sys.exit(1)

        # Use relative path for add_path_prefix if not provided
        if add_path_prefix is None:
            add_path_prefix = os.path.dirname(db)

        # Start server
        run_server(
            db_path=db,
            prefix=add_path_prefix,
            trim_prefix=remove_path_prefix,
            rerank_model=rerank_model,
            tool_name=tool_name,
            tool_description=tool_description,
            transport=transport,
            host=host,
            port=port,
            path=path
        )

    except Exception as e:
        console.print(f"[red]Error starting server: {str(e)}[/red]")
        sys.exit(1)


def perform_search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    top_k: int,
    add_path_prefix: Optional[str],
    remove_path_prefix: Optional[str],
    rerank: bool,
    embedding_model: str,
    rerank_model: str,
    console: Console,
) -> None:
    """Perform search and display results.

    Args:
        conn: DuckDB connection
        query: Search query
        top_k: Number of results to return
        add_path_prefix: Prefix to add to file paths in results
        remove_path_prefix: Prefix to remove from file paths in results
        rerank: Whether to rerank results
        embedding_model: Embedding model name
        rerank_model: Reranking model name
        console: Rich console for output
    """
    # Search for documents
    from duckdb_hybrid_document_search.searcher import search as searcher_search

    results = searcher_search(
        conn=conn,
        query=query,
        top_k=top_k,
        add_path_prefix=add_path_prefix,
        remove_path_prefix=remove_path_prefix,
        rerank=rerank,
    )

    # Print results
    console.print(f"[bold]Search results for: [cyan]{query}[/cyan][/bold]")
    console.print()

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, result in enumerate(results, 1):
        console.print(f"[bold]{i}. [green]{result['file_path']}[/green][/bold]")
        if result["header_path"]:
            console.print(f"   Section: {result['header_path']}")
        console.print(f"   Lines: {result['line_start']}-{result['line_end']}")
        console.print(f"   Score: {result['score']:.4f}")
        console.print()
        console.print(f"   [yellow]{result['content'][:200]}...[/yellow]")
        console.print()


@app.command()
def search(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search query (if not provided, interactive mode is enabled)",
    ),
    db: str = typer.Option(
        "index.duckdb",
        "--db",
        "-d",
        help="Path to DuckDB database file",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return",
        min=1,
        max=100,
    ),
    add_path_prefix: Optional[str] = typer.Option(
        None,
        "--add-path-prefix",
        "-p",
        help="Prefix to add to file paths in search results",
    ),
    remove_path_prefix: Optional[str] = typer.Option(
        None,
        "--remove-path-prefix",
        "-t",
        help="Prefix to remove from file paths in search results",
    ),
    rerank_model: str = typer.Option(
        "cl-nagoya/ruri-v3-reranker-310m",
        "--rerank-model",
        "-r",
        help="Hugging Face model ID for reranking",
    ),
    no_rerank: bool = typer.Option(
        False,
        "--no-rerank",
        help="Disable reranking of search results",
    ),
):
    """Search for documents using hybrid search.

    If no query is provided, interactive mode is enabled.
    """
    try:
        if not os.path.exists(db):
            console.print(f"[red]Error: Database file {db} not found[/red]")
            sys.exit(1)

        # Use relative path for add_path_prefix if not provided
        if add_path_prefix is None:
            add_path_prefix = os.path.dirname(db)

        # Connect to database
        conn = init_db(db, read_only=True)

        # Get embedding model from settings
        model_row = conn.execute(
            "SELECT value FROM settings WHERE key = 'embedding_model'"
        ).fetchone()

        if not model_row:
            console.print(
                f"[red]Error: 'embedding_model' not found in DB. Run the index command first.[/red]"
            )
            sys.exit(1)

        embedding_model = model_row[0]

        # Initialize models
        init_models(embedding_model, rerank_model)

        # Interactive mode if no query is provided
        if query is None:
            console.print("[bold]Interactive search mode[/bold] (Ctrl+C to exit)")
            console.print(f"Using database: [green]{db}[/green]")
            console.print(f"Embedding model: [green]{embedding_model}[/green]")
            console.print(
                f"Reranking: [green]{'Disabled' if no_rerank else f'Enabled ({rerank_model})'}[/green]"
            )
            console.print()

            while True:
                try:
                    # Get query from user
                    query = typer.prompt("Enter search query")

                    if not query.strip():
                        console.print("[yellow]Please enter a valid query[/yellow]")
                        continue

                    # Perform search
                    perform_search(
                        conn=conn,
                        query=query,
                        top_k=top_k,
                        add_path_prefix=add_path_prefix,
                        remove_path_prefix=remove_path_prefix,
                        rerank=not no_rerank,
                        embedding_model=embedding_model,
                        rerank_model=rerank_model,
                        console=console,
                    )

                    # Add a blank line between searches
                    console.print()

                except KeyboardInterrupt:
                    console.print("\n[bold]Exiting interactive mode[/bold]")
                    break
        else:
            # Single search mode
            perform_search(
                conn=conn,
                query=query,
                top_k=top_k,
                add_path_prefix=add_path_prefix,
                remove_path_prefix=remove_path_prefix,
                rerank=not no_rerank,
                embedding_model=embedding_model,
                rerank_model=rerank_model,
                console=console,
            )

    except Exception as e:
        console.print(f"[red]Error during search: {str(e)}[/red]")
        sys.exit(1)


@app.command()
def download_models(
    models: List[str] = typer.Argument(
        ...,
        help="Hugging Face model IDs to download (can specify multiple models)",
    ),
):
    """Pre-download and cache models to avoid delay on first search.

    This command downloads and caches the specified models so they don't need to be
    downloaded on first use. You can specify multiple models by separating them with spaces.

    Example:
        duckdb-hybrid-doc-search download-models cl-nagoya/ruri-v3-310m cl-nagoya/ruri-v3-reranker-310m
    """
    try:
        for model in models:
            console.print(f"Downloading model: [bold]{model}[/bold]")

            # Try to determine if this is an embedding model or a reranker model
            # This is a simple heuristic based on model name
            if "reranker" in model.lower():
                console.print("Detected as a reranker model")
                get_reranker_model(model)
                console.print(f"[green]Successfully downloaded and cached reranker model: {model}[/green]")
            else:
                console.print("Detected as an embedding model")
                get_embedding_model(model)
                console.print(f"[green]Successfully downloaded and cached embedding model: {model}[/green]")

        console.print(f"[green]All {len(models)} models have been successfully downloaded and cached[/green]")

    except Exception as e:
        console.print(f"[red]Error downloading models: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
