"""Embedding engine — supports Ollama (local) with FastEmbed fallback."""

import logging
from typing import Any

import httpx

from ..config import config

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_embedder: Any = None
_embedding_provider: str = ""


def _get_embedder():
    """Get or create the embedding model instance."""
    global _embedder, _embedding_provider
    if _embedder is not None:
        return _embedder

    cfg = config.models.embedding
    provider = cfg.provider

    if provider == "fastembed":
        try:
            from fastembed import TextEmbedding
            model_name = cfg.model
            logger.info(f"Loading FastEmbed model: {model_name}")
            _embedder = TextEmbedding(model_name=model_name)
            _embedding_provider = "fastembed"
            return _embedder
        except Exception as e:
            logger.warning(f"FastEmbed unavailable ({e}), falling back to Ollama")

    # Default / fallback: use Ollama
    logger.info(f"Using Ollama embedding: {cfg.model}")
    _embedder = True  # Sentinel value
    _embedding_provider = "ollama"
    return _embedder


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is list[float])
    """
    if not texts:
        return []

    _get_embedder()

    if _embedding_provider == "fastembed":
        embeddings = list(_embedder.embed(texts))
        return [e.tolist() for e in embeddings]
    else:
        # Ollama embedding
        return await _embed_via_ollama(texts)


async def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text."""
    results = await embed_texts([text])
    return results[0] if results else []


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous version for use in non-async contexts."""
    if not texts:
        return []
    _get_embedder()
    if _embedding_provider == "fastembed":
        embeddings = list(_embedder.embed(texts))
        return [e.tolist() for e in embeddings]
    else:
        raise RuntimeError("Ollama embedding requires async context")


def embed_single_sync(text: str) -> list[float]:
    """Synchronous single-text embedding."""
    results = embed_texts_sync([text])
    return results[0] if results else []


async def _embed_via_ollama(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using Ollama API."""
    cfg = config.models.embedding
    url = f"{cfg.base_url}/api/embed"
    vectors: list[list[float]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for text in texts:
            try:
                resp = await client.post(
                    url,
                    json={"model": cfg.model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                vectors.append(data["embeddings"][0])
            except Exception as e:
                logger.warning(f"Ollama embedding failed for text '{text[:50]}...': {e}")
                # Return zero vector as fallback
                vec_size = config.storage.qdrant.vector_size
                vectors.append([0.0] * vec_size)

    return vectors
