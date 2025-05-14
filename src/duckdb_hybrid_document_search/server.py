"""MCP stdio server implementation."""

import sys
from typing import Dict, List, Optional, Union

import duckdb
from mcp.server.fastmcp import FastMCP

from duckdb_hybrid_document_search.db import init_db
from duckdb_hybrid_document_search.searcher import init_models, search
from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)

def run_server(
    db_path: str,
    prefix: str,
    rerank_model: str,
    tool_name: str = "search_documents",
    tool_description: str = "Search for local documents"
) -> None:
    """Run MCP stdio server.

    Args:
        db_path: Path to DuckDB database
        prefix: Prefix to add to file paths in search results
        rerank_model: Hugging Face model ID for reranking
        tool_name: Name of the MCP tool (default: "search_documents")
        tool_description: Description of the MCP tool (default: "Search for local documents")
    """
    logger.info(f"Starting MCP server with database: {db_path}")
    logger.info(f"Using tool name: {tool_name}")
    logger.info(f"Using tool description: {tool_description}")

    # Connect to database
    conn = init_db(db_path, read_only=True)

    # Get embedding model from settings
    model_row = conn.execute("SELECT value FROM settings WHERE key = 'embedding_model'").fetchone()

    if not model_row:
        logger.error("'embedding_model' not found in database settings")
        sys.exit("ERROR: 'embedding_model' not found in DB. Run the index command first.")

    embedding_model = model_row[0]
    logger.info(f"Using embedding model: {embedding_model}")
    logger.info(f"Using reranker model: {rerank_model}")

    init_models(embedding_model, rerank_model)

    mcp = FastMCP("duckdb-hybrid-doc-search")

    @mcp.tool(name=tool_name, description=tool_description)
    def search_documents(
        query: str,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """Search for documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dictionary with search results
        """
        logger.info(f"MCP search request: {query} (top_k={top_k})")

        results = search(
            conn=conn,
            query=query,
            top_k=top_k,
            file_path_prefix=prefix,
            rerank=True,
        )

        return {"results": results}

    # Run server
    logger.info("Starting MCP stdio server")
    mcp.run(transport="stdio")
