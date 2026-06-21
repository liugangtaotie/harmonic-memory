"""Semantic deduplication via vector similarity."""

import logging

from ..config import config
from .embedder import embed_single

logger = logging.getLogger(__name__)


async def is_duplicate(
    content: str,
    qdrant_client,
    threshold: float | None = None,
) -> tuple[bool, str | None]:
    """Check if a memory is semantically similar to any existing memory.

    Args:
        content: The new memory content to check
        qdrant_client: MemoryQdrant instance
        threshold: Cosine similarity threshold (default from config)

    Returns:
        (is_duplicate, existing_memory_id_or_None)
    """
    if threshold is None:
        threshold = config.ingestion.dedup.vector_threshold

    # Generate embedding for the new content
    embedding = await embed_single(content)
    if not embedding:
        return False, None

    # Search for near-duplicates in Qdrant
    results = await qdrant_client.search(
        vector=embedding,
        limit=3,
        score_threshold=threshold,
    )

    if results:
        best = results[0]
        logger.info(
            f"Dedup: found similar memory (score={best['score']:.3f}, "
            f"id={best['id']})"
        )
        return True, best["id"]

    return False, None


async def find_similar(
    content: str,
    qdrant_client,
    limit: int = 5,
    threshold: float = 0.75,
) -> list[dict]:
    """Find semantically similar memories (for consolidation/reference).

    Returns list of {id, score, payload} dicts.
    """
    embedding = await embed_single(content)
    if not embedding:
        return []

    return await qdrant_client.search(
        vector=embedding,
        limit=limit,
        score_threshold=threshold,
    )
