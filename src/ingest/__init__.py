"""Ingestion pipeline — extract, classify, embed, score, dedup, store."""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ..config import config
from ..db.sqlite import MemoryDB
from ..db.qdrant_client import MemoryQdrant
from .extractor import extract_memories
from .classifier import classify_batch
from .embedder import embed_texts
from .scorer import quality_score, importance_estimate
from .dedup import is_duplicate

logger = logging.getLogger(__name__)


async def ingest(
    text: str,
    source: str = "unknown",
    source_ref: str | None = None,
    db: MemoryDB | None = None,
    qdrant: MemoryQdrant | None = None,
    use_fallback: bool = False,
) -> dict[str, Any]:
    """Full ingestion pipeline: text → extracted memories → stored with vectors.

    Args:
        text: Raw text to process
        source: Source identifier (claude, codex, hermes, manual)
        source_ref: Specific reference (session ID, file path)
        db: MemoryDB instance (created if None)
        qdrant: MemoryQdrant instance (created if None)
        use_fallback: Force DeepSeek fallback

    Returns:
        Dict with ingestion summary
    """
    if db is None:
        db = MemoryDB()
        db.init_schema()
    if qdrant is None:
        qdrant = MemoryQdrant()

    start_time = time.time()
    log_id = db.log_ingestion(
        source=source,
        source_ref=source_ref,
        raw_bytes=len(text.encode("utf-8")),
        status="processing",
        metadata={"use_fallback": use_fallback},
    )

    # Stage 1: Extract memories via LLM
    extracted = await extract_memories(
        text,
        source=source,
        source_ref=source_ref,
        use_fallback=use_fallback,
    )

    if not extracted:
        db.update_ingestion(log_id, "success", extracted_count=0)
        return {
            "status": "success",
            "memories_created": 0,
            "duplicates_skipped": 0,
            "rejected_low_quality": 0,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    # Stage 2: Classify and score
    classify_batch(extracted)
    for m in extracted:
        if not m.get("importance") or m["importance"] == 0.3:  # Default
            m["importance"] = importance_estimate(m["content"], m["type"])
        m["quality"] = quality_score(m)
        m["summary"] = m["content"][:200]  # First 200 chars as summary

    # Stage 3: Generate embeddings
    contents = [m["content"] for m in extracted]
    embeddings = await embed_texts(contents)

    # Stage 4: Dedup and store
    created = 0
    duplicates = 0
    rejected = 0
    memory_ids = []

    for i, mem in enumerate(extracted):
        # Check quality
        if mem.get("quality", 0) < config.ingestion.quality.min_confidence:
            rejected += 1
            continue

        # Dedup check
        is_dup, existing_id = await is_duplicate(mem["content"], qdrant)
        if is_dup:
            duplicates += 1
            # Update access count on existing memory
            if existing_id:
                db.record_access(existing_id)
            continue

        # Store vector
        import uuid
        embedding_id = await qdrant.upsert(
            point_id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "type": mem["type"],
                "source": source,
                "content_preview": mem["content"][:100],
            },
        )

        # Store in SQLite
        memory_dict = {
            "type": mem["type"],
            "content": mem["content"],
            "summary": mem.get("summary", mem["content"][:200]),
            "source": source,
            "source_ref": source_ref,
            "confidence": mem.get("confidence", 0.5),
            "importance": mem.get("importance", 0.5),
            "state": "extracted",
            "embedding_id": embedding_id,
            "metadata": {
                "keywords": mem.get("keywords", []),
                "quality": mem.get("quality", 0.5),
                "extraction_provider": mem.get("extraction_provider", "unknown"),
                "extraction_latency_ms": mem.get("extraction_latency_ms", 0),
            },
        }
        memory_id = db.insert_memory(memory_dict)
        memory_ids.append(memory_id)
        created += 1

    total_ms = int((time.time() - start_time) * 1000)
    db.update_ingestion(log_id, "success", extracted_count=created)

    # Auto-promote high-confidence memories to "active"
    for mid in memory_ids:
        mem = db.get_memory(mid)
        if mem and mem["confidence"] >= 0.7 and mem["importance"] >= 0.5:
            db.update_memory(mid, {"state": "active"})

    result = {
        "status": "success",
        "memories_created": created,
        "duplicates_skipped": duplicates,
        "rejected_low_quality": rejected,
        "memory_ids": memory_ids,
        "latency_ms": total_ms,
    }

    logger.info(
        f"Ingestion complete: {created} created, {duplicates} dupes, "
        f"{rejected} rejected in {total_ms}ms"
    )

    return result
