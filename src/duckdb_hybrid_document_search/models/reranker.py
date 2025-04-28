"""CrossEncoder reranker implementation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)

# Global cache for reranker models
_reranker_models: Dict[str, CrossEncoder] = {}


def get_reranker_model(model_name: str) -> CrossEncoder:
    """Get or load a reranker model.

    Args:
        model_name: Hugging Face model ID

    Returns:
        CrossEncoder model
    """
    global _reranker_models

    if model_name in _reranker_models:
        logger.debug(f"Using cached reranker model: {model_name}")
        return _reranker_models[model_name]

    logger.info(f"Loading reranker model: {model_name}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = CrossEncoder(model_name, device=device)

    # Cache model
    _reranker_models[model_name] = model

    return model


def rerank_results(
    model_name: str,
    query: str,
    texts: List[str],
    batch_size: int = 8,
) -> List[Tuple[int, float]]:
    """Rerank search results using a CrossEncoder.

    Args:
        model_name: Hugging Face model ID
        query: Search query
        texts: List of texts to rerank
        batch_size: Batch size for reranking

    Returns:
        List of (index, score) tuples sorted by score in descending order
    """
    model = get_reranker_model(model_name)

    logger.info(f"Reranking {len(texts)} results")

    # Truncate long texts to avoid token limit issues
    truncated_texts = [text[:2048] if len(text) > 2048 else text for text in texts]

    # Create query-document pairs
    pairs = [(query, text) for text in truncated_texts]

    # Generate scores
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    # Create (index, score) pairs and sort by score in descending order
    indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return indexed_scores
