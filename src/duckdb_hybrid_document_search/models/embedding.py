"""Embedding model loading and caching."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from duckdb_hybrid_document_search.utils.logging import get_logger

logger = get_logger(__name__)

# Global cache for embedding models
_embedding_models: Dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get or load an embedding model.

    Args:
        model_name: Hugging Face model ID

    Returns:
        SentenceTransformer model
    """
    global _embedding_models

    if model_name in _embedding_models:
        logger.debug(f"Using cached embedding model: {model_name}")
        return _embedding_models[model_name]

    logger.info(f"Loading embedding model: {model_name}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = SentenceTransformer(model_name, device=device)

    # Cache model
    _embedding_models[model_name] = model

    return model


def generate_embeddings(
    model_name: str,
    texts: List[str],
    batch_size: int = 8,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        model_name: Hugging Face model ID
        texts: List of texts to embed
        batch_size: Batch size for embedding generation
        show_progress_bar: Whether to show a progress bar

    Returns:
        Array of embeddings
    """
    model = get_embedding_model(model_name)

    logger.info(f"Generating embeddings for {len(texts)} texts")

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    return embeddings
