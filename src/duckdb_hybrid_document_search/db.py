"""DuckDB connection and schema DDL."""

import logging
from typing import Optional

import duckdb
from ulid import ULID
from sentence_transformers import SentenceTransformer

from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)


def get_embedding_dim_from_model(model_name: str) -> int:
    """Get the embedding dimension of a Sentence Transformer model."""
    try:
        # Load the model just to get its dimension
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        if dim is None:
            # Fallback or error if dimension cannot be determined
            logger.warning(f"Could not determine embedding dimension for {model_name}. Falling back to default.")
            # Or raise an error: raise ValueError(f"Could not determine embedding dimension for {model_name}")
            return 384 # Or a sensible default / raise error
        logger.info(f"Determined embedding dimension for {model_name}: {dim}")
        # Clean up the model instance if possible, although SentenceTransformer might not have explicit cleanup.
        # Depending on the library, you might need `del model` and `torch.cuda.empty_cache()` if using GPU.
        del model
        # If using GPU:
        # import torch
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        return dim
    except Exception as e:
        logger.error(f"Failed to load model {model_name} to get dimension: {e}")
        raise ValueError(f"Failed to load model {model_name} to get dimension") from e


def init_db(
    db_path: str,
    read_only: bool = False,
    embedding_model: Optional[str] = None,
) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection and create schema if needed.

    Args:
        db_path: Path to DuckDB database file
        read_only: Whether to open in read-only mode
        embedding_model: Name of the embedding model (used to determine dimension if creating schema)

    Returns:
        DuckDB connection
    """
    logger.info(f"Connecting to DuckDB at {db_path} (read_only={read_only})")
    conn = duckdb.connect(db_path, read_only=read_only)

    # Create schema if not in read-only mode
    if not read_only:
        embedding_dim: Optional[int] = None
        # Try to get dimension from existing settings first if DB exists
        try:
            existing_dim_str = get_setting(conn, "embedding_dim")
            if existing_dim_str:
                embedding_dim = int(existing_dim_str)
                logger.info(f"Using existing embedding dimension from settings: {embedding_dim}")
                # Optional: Verify if embedding_model matches the stored one if provided
                stored_model = get_setting(conn, "embedding_model")
                if embedding_model and stored_model and embedding_model != stored_model:
                     logger.warning(f"Provided embedding model '{embedding_model}' differs from stored model '{stored_model}'. Using stored dimension {embedding_dim}.")
            elif embedding_model:
                # If no dimension stored, get it from the provided model name
                embedding_dim = get_embedding_dim_from_model(embedding_model)
            else:
                 # Fallback if DB is new and no model is provided during init
                 logger.warning("No embedding model provided and no dimension stored. Using default dimension 384.")
                 embedding_dim = 384

        except duckdb.CatalogException:
             # settings table probably doesn't exist yet (new DB)
             if embedding_model:
                 embedding_dim = get_embedding_dim_from_model(embedding_model)
             else:
                 logger.warning("Creating new DB without embedding model specified. Using default dimension 384.")
                 embedding_dim = 384 # Default for new DB without model info
        except Exception as e:
             logger.error(f"Error determining embedding dimension: {e}. Using default 384.")
             embedding_dim = 384 # Fallback on other errors


        create_schema(conn, embedding_dim=embedding_dim)
        # Store the dimension used
        if embedding_dim:
            store_setting(conn, "embedding_dim", str(embedding_dim))
        # Store the model name if provided
        if embedding_model:
             store_setting(conn, "embedding_model", embedding_model)


    return conn


def create_schema(conn: duckdb.DuckDBPyConnection, embedding_dim: int = 384) -> None:
    """Create the database schema.

    Args:
        conn: DuckDB connection
        embedding_dim: The dimension for the embedding vector.
    """
    logger.info(f"Creating database schema with embedding dimension: {embedding_dim}")

    # Create documents table with dynamic embedding dimension
    conn.execute(
        f"""
    CREATE TABLE IF NOT EXISTS documents (
      doc_id      TEXT PRIMARY KEY,
      file_path   TEXT NOT NULL,
      header_path TEXT,
      line_start  INTEGER,
      line_end    INTEGER,
      content     TEXT,
      tokens      TEXT,
      embedding   FLOAT[{embedding_dim}]
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
