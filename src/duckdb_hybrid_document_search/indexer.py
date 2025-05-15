"""Document indexer implementation."""

import logging
from typing import List

import duckdb
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn
from ulid import ULID

from duckdb_hybrid_document_search.db import clear_documents
from duckdb_hybrid_document_search.models.embedding import generate_embeddings
from duckdb_hybrid_document_search.splitter import Chunk, SplitterType, create_splitter
from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)


def index_directories(
    conn: duckdb.DuckDBPyConnection,
    directories: List[str],
    embedding_model: str,
    workers: int = 4,
    clear: bool = False,
    splitter_type: SplitterType = SplitterType.LLAMA_INDEX,
    path_prefix_to_trim: str = None,
) -> None:
    """Index Markdown documents in directories.

    Args:
        conn: DuckDB connection
        directories: List of directories to index
        embedding_model: Hugging Face model ID for embeddings
        workers: Number of worker processes
        clear: Whether to clear existing documents
        splitter_type: Type of splitter to use (CHONKIE or LLAMA_INDEX)
        path_prefix_to_trim: Prefix to trim from file paths (e.g., '/app/')
    """
    logger.info(f"Indexing directories: {', '.join(directories)}")

    # Clear existing documents if requested
    if clear:
        clear_documents(conn)

    # Create splitter using factory function
    splitter = create_splitter(splitter_type=splitter_type)
    logger.info(f"Using splitter: {splitter_type.name} ({splitter.__class__.__name__})")

    # Split documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        # Add a task with an unknown total initially
        task_id = progress.add_task("Scanning directories...", total=None)

        # Define a callback function to update progress
        def update_progress(completed, total):
            # Update the total if it's the first update
            if progress.tasks[task_id].total is None and total > 0:
                progress.update(task_id, total=total, description=f"Splitting documents... [0/{total}]")

            # Update the progress
            if total > 0:
                progress.update(task_id, completed=completed,
                               description=f"Splitting documents... [{completed}/{total}]")

        # Call split_directory with the progress callback
        chunks = splitter.split_directory(directories, workers=workers, progress_callback=update_progress)

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

        # Process file path
        file_path = chunk.file_path

        # Trim specified prefix if provided
        if path_prefix_to_trim and file_path.startswith(path_prefix_to_trim):
            file_path = file_path[len(path_prefix_to_trim):]

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
