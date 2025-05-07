"""Document indexer implementation."""

import logging
from typing import List

import duckdb
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn
from ulid import ULID

from duckdb_hybrid_document_search.db import clear_documents
from duckdb_hybrid_document_search.models.embedding import generate_embeddings
from duckdb_hybrid_document_search.splitter import Chunk, MarkdownSplitter
from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)


def index_directories(
    conn: duckdb.DuckDBPyConnection,
    directories: List[str],
    embedding_model: str,
    workers: int = 4,
    clear: bool = False,
) -> None:
    """Index Markdown documents in directories.

    Args:
        conn: DuckDB connection
        directories: List of directories to index
        embedding_model: Hugging Face model ID for embeddings
        workers: Number of worker processes
        clear: Whether to clear existing documents
    """
    logger.info(f"Indexing directories: {', '.join(directories)}")

    # Clear existing documents if requested
    if clear:
        clear_documents(conn)

    # Create splitter
    splitter = MarkdownSplitter()

    # Split documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        progress.add_task("Splitting documents...", total=None)
        chunks = splitter.split_directory(directories, workers=workers)

    if not chunks:
        logger.warning("No documents found to index")
        return

    logger.info(f"Found {len(chunks)} chunks to index")

    # Generate embeddings
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        progress.add_task("Generating embeddings...", total=None)
        embeddings = generate_embeddings(
            embedding_model,
            [chunk.content for chunk in chunks],
            show_progress_bar=False,
        )

    # Insert chunks into database
    logger.info("Inserting chunks into database")

    # Prepare data for batch insert
    data = []
    for i, chunk in enumerate(chunks):
        # Generate a ULID for each document
        doc_id = str(ULID())

        # Remove leading slash from file path if present
        file_path = chunk.file_path
        if file_path.startswith('/'):
            file_path = file_path[1:]

        data.append(
            {
                "doc_id": doc_id,
                "file_path": file_path,
                "header_path": chunk.header_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "content": chunk.content,
                "tokens": chunk.tokens,
                "embedding": embeddings[i].tolist(),
            }
        )

    # Batch insert
    conn.executemany(
        """
    INSERT INTO documents (
        doc_id, file_path, header_path, line_start, line_end, content, tokens, embedding
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?
    )
    """,
        [
            (
                d["doc_id"],
                d["file_path"],
                d["header_path"],
                d["line_start"],
                d["line_end"],
                d["content"],
                d["tokens"],
                d["embedding"],
            )
            for d in data
        ],
    )

    logger.info(f"Indexed {len(chunks)} chunks successfully")
