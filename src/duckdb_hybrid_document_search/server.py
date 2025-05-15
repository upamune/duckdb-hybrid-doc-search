"""MCP server implementation with support for STDIO and HTTP transports."""

import sys
from typing import Dict, List, Literal, Optional, Union

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
    tool_description: str = "Search for local documents",
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp",
) -> None:
    """Run MCP server with specified transport.

    Args:
        db_path: Path to DuckDB database
        prefix: Prefix to add to file paths in search results
        rerank_model: Hugging Face model ID for reranking
        tool_name: Name of the MCP tool (default: "search_documents")
        tool_description: Description of the MCP tool (default: "Search for local documents")
        transport: Transport protocol to use (default: "stdio")
        host: Host to bind to for HTTP transport (default: "127.0.0.1")
        port: Port to bind to for HTTP transport (default: 8765)
        path: Path for streamable-http transport (default: "/mcp")
    """
    logger.info(f"Starting MCP server with database: {db_path}")
    logger.info(f"Using tool name: {tool_name}")
    logger.info(f"Using tool description: {tool_description}")
    logger.info(f"Using transport: {transport}")

    if transport == "streamable-http":
        logger.info(f"HTTP server will be available at: {host}:{port}")
        logger.info(f"Streamable HTTP endpoint: {path}")

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

    mcp = FastMCP(name="duckdb-hybrid-doc-search")
    if transport == "streamable-http":
        mcp = FastMCP(name="duckdb-hybrid-doc-search", host=host, port=port, path=path)

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

    # Run server with specified transport
    if transport == "stdio":
        logger.info("Starting MCP stdio server")
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        logger.info(f"Starting MCP streamable-http server on {host}:{port}{path}")
        mcp.run(transport="streamable-http")
