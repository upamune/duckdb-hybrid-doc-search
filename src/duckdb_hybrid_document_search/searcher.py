"""Hybrid search and reranking implementation."""

import logging
import os
from typing import Dict, List, Optional, Union

import duckdb
import numpy as np
from lindera_py import Segmenter, Tokenizer, load_dictionary

from duckdb_hybrid_document_search.models.embedding import generate_embeddings, get_embedding_model
from duckdb_hybrid_document_search.models.reranker import rerank_results, get_reranker_model
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

    # Preload models to avoid delay on first search
    logger.info("Preloading embedding model...")
    get_embedding_model(embedding_model)

    logger.info("Preloading reranker model...")
    get_reranker_model(reranker_model)


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

    logger.info(f"FTS search found {len(fts_results)} results")
    if fts_results:
        logger.debug(f"Top FTS results: {fts_results[:3]}")

    # VSS search
    try:
        # Convert embedding to list for SQL compatibility
        embedding_list = embedding_q.tolist()

        # Perform VSS search using HNSW index
        vss_results = conn.execute(
            """
        SELECT doc_id, vss_hnsw_documents.cosine_similarity(embedding, ?) AS score
        FROM documents
        ORDER BY score DESC
        LIMIT ?
        """,
            [embedding_list, top_k],
        ).fetchall()

        logger.info(f"VSS search found {len(vss_results)} results")
        if vss_results:
            logger.debug(f"Top VSS results: {vss_results[:3]}")
    except Exception as e:
        logger.error(f"VSS search failed: {e}")
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

    # Create score maps for FTS and VSS results
    fts_scores = {doc_id: score for doc_id, score in fts_results}
    vss_scores = {doc_id: score for doc_id, score in vss_results}

    # Convert to dictionaries
    results = []
    for doc in documents:
        doc_id, file_path, header_path, line_start, line_end, content = doc

        # Add file path prefix if provided
        if file_path_prefix:
            # Use the full relative path, not just the basename
            full_path = os.path.join(file_path_prefix, file_path)
        else:
            full_path = file_path

        # Calculate hybrid score (combine FTS and VSS scores)
        fts_score = fts_scores.get(doc_id, 0.0)
        vss_score = vss_scores.get(doc_id, 0.0)

        # Normalize and combine scores (simple average for now)
        # We could use a weighted average if one method is more reliable
        hybrid_score = 0.0
        if doc_id in fts_scores and doc_id in vss_scores:
            # Both methods found this document
            hybrid_score = (fts_score + vss_score) / 2.0
        elif doc_id in fts_scores:
            # Only FTS found this document
            hybrid_score = fts_score * 0.8  # Slightly reduce score for single-method matches
        elif doc_id in vss_scores:
            # Only VSS found this document
            hybrid_score = vss_score * 0.8  # Slightly reduce score for single-method matches

        results.append(
            {
                "doc_id": doc_id,
                "file_path": full_path,
                "header_path": header_path,
                "line_start": line_start,
                "line_end": line_end,
                "content": content,
                "score": hybrid_score,  # Initial score before reranking
                "fts_score": fts_score,
                "vss_score": vss_score,
            }
        )

    # Sort by initial hybrid score before reranking
    results.sort(key=lambda x: x["score"], reverse=True)

    # Rerank results if requested
    if rerank and _reranker_model and results:
        # Log initial scores for debugging
        logger.debug(f"Initial scores before reranking: {[(r['doc_id'], r['score']) for r in results[:5]]}")

        reranked = rerank_results(
            _reranker_model,
            query,
            [r["content"] for r in results],
        )

        # Update scores and sort results
        for i, (idx, score) in enumerate(reranked):
            # Store original score for debugging
            results[idx]["original_score"] = results[idx]["score"]
            # Update with reranker score
            results[idx]["score"] = float(score)

        # Log reranked scores for debugging
        logger.debug(f"Reranked scores: {[(r['doc_id'], r['score']) for r in sorted(results[:5], key=lambda x: x['score'], reverse=True)]}")

        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)

    # Filter out results with very low scores (threshold can be adjusted)
    score_threshold = 0.01  # Minimum acceptable score
    filtered_results = [r for r in results if r["score"] > score_threshold]

    if len(filtered_results) < len(results):
        logger.info(f"Filtered out {len(results) - len(filtered_results)} results with scores below {score_threshold}")

    # Limit to top_k
    filtered_results = filtered_results[:top_k]

    return filtered_results
