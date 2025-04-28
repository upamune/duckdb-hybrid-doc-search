"""Hybrid search and reranking implementation."""

import logging
import os
from typing import Dict, List, Optional, Union

import duckdb
import numpy as np
from lindera_py import Segmenter, Tokenizer, load_dictionary

from duckdb_hybrid_document_search.models.embedding import generate_embeddings
from duckdb_hybrid_document_search.models.reranker import rerank_results
from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)

# Global tokenizer instance
_tokenizer: Optional[Tokenizer] = None

# Global embedding model name
_embedding_model: Optional[str] = None

# Global reranker model name
_reranker_model: Optional[str] = None


def init_models(embedding_model: str, reranker_model: str) -> None:
    """Initialize models for search.

    Args:
        embedding_model: Hugging Face model ID for embeddings
        reranker_model: Hugging Face model ID for reranking
    """
    global _tokenizer, _embedding_model, _reranker_model

    logger.info(f"Initializing models: {embedding_model}, {reranker_model}")

    # Initialize tokenizer using the official API
    dictionary = load_dictionary("ipadic")
    segmenter = Segmenter("normal", dictionary)
    _tokenizer = Tokenizer(segmenter)

    # Set model names
    _embedding_model = embedding_model
    _reranker_model = reranker_model


def tokenize_query(query: str) -> str:
    """Tokenize a query using Lindera.

    Args:
        query: Query to tokenize

    Returns:
        Space-joined tokens
    """
    global _tokenizer

    if _tokenizer is None:
        try:
            _tokenizer = Tokenizer(dict_type="ipadic")
        except TypeError:
            _tokenizer = Tokenizer(dictionary_type="ipadic")

    tokens = _tokenizer.tokenize(query)
    return " ".join(token.text for token in tokens)


def search(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    top_k: int = 5,
    file_path_prefix: Optional[str] = None,
    rerank: bool = True,
) -> List[Dict[str, Union[str, int, float]]]:
    """Search for documents using hybrid search.

    Args:
        conn: DuckDB connection
        query: Search query
        top_k: Number of results to return
        file_path_prefix: Prefix to add to file paths in results
        rerank: Whether to rerank results

    Returns:
        List of search results
    """
    global _embedding_model, _reranker_model

    if _embedding_model is None:
        raise ValueError("Embedding model not initialized. Call init_models first.")

    logger.info(f"Searching for: {query}")

    # Tokenize query
    tokens_q = tokenize_query(query)
    logger.debug(f"Tokenized query: {tokens_q}")

    # Generate query embedding
    embedding_q = generate_embeddings(_embedding_model, [query], show_progress_bar=False)[0]

    # FTS search
    fts_results = conn.execute(
        """
    SELECT doc_id, fts_main_documents.match_bm25(doc_id, ?) AS score
    FROM documents
    ORDER BY score DESC
    LIMIT ?
    """,
        [tokens_q, top_k],
    ).fetchall()

    # VSS search
    # Skip VSS search for now due to type issues
    vss_results = []

    # Merge results
    doc_ids = set()
    merged_ids = []

    # Add FTS results
    for doc_id, _ in fts_results:
        if doc_id not in doc_ids:
            doc_ids.add(doc_id)
            merged_ids.append(doc_id)

    # Add VSS results
    for doc_id, _ in vss_results:
        if doc_id not in doc_ids:
            doc_ids.add(doc_id)
            merged_ids.append(doc_id)

    # Fetch documents
    if not merged_ids:
        return []

    placeholders = ", ".join(["?"] * len(merged_ids))
    documents = conn.execute(
        f"""
    SELECT doc_id, file_path, header_path, line_start, line_end, content
    FROM documents
    WHERE doc_id IN ({placeholders})
    """,
        merged_ids,
    ).fetchall()

    # Convert to dictionaries
    results = []
    for doc in documents:
        doc_id, file_path, header_path, line_start, line_end, content = doc

        # Add file path prefix if provided
        if file_path_prefix:
            # 相対パスを使用する場合、ファイル名だけを取得して結合
            full_path = os.path.join(file_path_prefix, os.path.basename(file_path))
        else:
            full_path = file_path

        results.append(
            {
                "doc_id": doc_id,
                "file_path": full_path,
                "header_path": header_path,
                "line_start": line_start,
                "line_end": line_end,
                "content": content,
                "score": 0.0,  # Will be updated by reranking
            }
        )

    # Rerank results if requested
    if rerank and _reranker_model:
        reranked = rerank_results(
            _reranker_model,
            query,
            [r["content"] for r in results],
        )

        # Update scores and sort results
        for i, (idx, score) in enumerate(reranked):
            results[idx]["score"] = float(score)

        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k
        results = results[:top_k]

    return results
