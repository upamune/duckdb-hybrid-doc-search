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
    add_path_prefix: Optional[str] = None,
    remove_path_prefix: Optional[str] = None,
    rerank: bool = True,
) -> List[Dict[str, Union[str, int, float]]]:
    """Search for documents using hybrid search.

    Args:
        conn: DuckDB connection
        query: Search query
        top_k: Number of results to return
        add_path_prefix: Prefix to add to file paths in results
        remove_path_prefix: Prefix to remove from file paths in results
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
        # Get embedding dimension from the list
        embedding_dim = len(embedding_list)

        vss_results = conn.execute(
            f"""
        SELECT doc_id, array_cosine_distance(embedding, ?::FLOAT[{embedding_dim}]) AS score
        FROM documents
        ORDER BY score ASC
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

        # Process file path: first trim prefix if specified, then add prefix if specified
        processed_path = file_path

        # 1. Remove path prefix if provided
        if remove_path_prefix:
            # Normalize both paths to ensure consistent format
            norm_file_path = os.path.normpath(processed_path)
            norm_prefix = os.path.normpath(remove_path_prefix)

            # Check if the normalized file path starts with the normalized prefix
            if norm_file_path.startswith(norm_prefix):
                # Remove the prefix
                processed_path = norm_file_path[len(norm_prefix):]
                # Ensure the path starts with a slash if it's not empty
                if processed_path and not processed_path.startswith('/'):
                    processed_path = '/' + processed_path
                # Remove leading slash if present (to get a relative path)
                if processed_path.startswith('/'):
                    processed_path = processed_path[1:]

        # 2. Add path prefix if provided
        if add_path_prefix:
            # Use the full relative path, not just the basename
            full_path = os.path.join(add_path_prefix, processed_path)
        else:
            full_path = processed_path

        # Calculate hybrid score (combine FTS and VSS scores)
        # Get the score from the map; it could be None if the DB returned NULL for score
        _fts_score_from_map = fts_scores.get(doc_id)
        _vss_score_from_map = vss_scores.get(doc_id)

        # Ensure scores are float, defaulting to 0.0 if None
        fts_score_val = _fts_score_from_map if _fts_score_from_map is not None else 0.0
        vss_score_val = _vss_score_from_map if _vss_score_from_map is not None else 0.0

        # Determine if the document was found by each method (i.e., score was not NULL and doc_id was present)
        doc_found_by_fts = _fts_score_from_map is not None
        doc_found_by_vss = _vss_score_from_map is not None

        hybrid_score = 0.0
        if doc_found_by_fts and doc_found_by_vss:
            # Both methods found this document with a valid score
            hybrid_score = (fts_score_val + vss_score_val) / 2.0
        elif doc_found_by_fts:
            # Only FTS found this document with a valid score
            hybrid_score = fts_score_val * 0.8
        elif doc_found_by_vss:
            # Only VSS found this document with a valid score
            hybrid_score = vss_score_val * 0.8
        # If neither method found the document with a valid score, hybrid_score remains 0.0.

        results.append(
            {
                "doc_id": doc_id,
                "file_path": full_path,
                "header_path": header_path,
                "line_start": line_start,
                "line_end": line_end,
                "content": content,
                "score": hybrid_score,
                "fts_score": fts_score_val,
                "vss_score": vss_score_val,
            }
        )

    # Sort by initial hybrid score before reranking
    results.sort(key=lambda x: x["score"], reverse=True)

    # Rerank results if requested
    if rerank and results:
        # Log initial scores for debugging
        logger.debug(f"Initial scores before reranking: {[(r['doc_id'], r['score']) for r in results[:5]]}")

        # Check if reranker model is available
        if _reranker_model is None:
            logger.warning("Reranker model is None. Skipping reranking.")
        else:
            try:
                # Make a copy of the original scores in case reranking fails
                for r in results:
                    r["original_score"] = r["score"]

                # Perform reranking
                reranked = rerank_results(
                    _reranker_model,
                    query,
                    [r["content"] for r in results],
                )

                # Only update scores if reranking was successful
                if reranked and len(reranked) > 0:
                    # Update scores and sort results
                    for i, (idx, score) in enumerate(reranked):
                        if idx < len(results):  # Ensure index is valid
                            # Update with reranker score
                            if score is not None:
                                try:
                                    results[idx]["score"] = float(score)
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Could not convert reranker score to float: {score}, error: {e}")
                                    # Keep original score
                                    results[idx]["score"] = results[idx]["original_score"]
                            else:
                                # Keep original score if reranker score is None
                                results[idx]["score"] = results[idx]["original_score"]

                    # Log reranked scores for debugging
                    logger.debug(f"Reranked scores: {[(r['doc_id'], r['score']) for r in sorted(results[:5], key=lambda x: x['score'], reverse=True)]}")

                    # Sort by score in descending order
                    results.sort(key=lambda x: x["score"], reverse=True)
                else:
                    logger.warning("Reranking returned empty results. Using original scores.")
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                # Restore original scores if reranking fails
                for r in results:
                    if "original_score" in r:
                        r["score"] = r["original_score"]

    # Filter out results with very low scores (threshold can be adjusted)
    score_threshold = 0.01  # Minimum acceptable score
    filtered_results = [r for r in results if r["score"] > score_threshold]

    if len(filtered_results) < len(results):
        logger.info(f"Filtered out {len(results) - len(filtered_results)} results with scores below {score_threshold}")

    # Limit to top_k
    filtered_results = filtered_results[:top_k]

    return filtered_results
