#!/usr/bin/env python3
"""Backfill memory edges for all existing memories.

For each memory, searches Qdrant for its vector neighbors
and creates edges in the memory_edges table.

Run: cd /f/harmonic-memory && uv run python scripts/backfill_edges.py
"""

import asyncio
import time
from src.db.sqlite import MemoryDB
from src.db.qdrant_client import MemoryQdrant
from src.ingest.embedder import embed_texts


def _relation_type(score: float) -> str:
    if score >= 0.92:
        return "extends"
    elif score >= 0.85:
        return "supports"
    else:
        return "references"


async def backfill(batch_size: int = 50, threshold: float = 0.75, max_links: int = 5):
    db = MemoryDB()
    db.init_schema()
    qdrant = MemoryQdrant()

    # Get all active/extracted memories
    rows = db.conn.execute(
        "SELECT id, content FROM memories WHERE state IN ('active','extracted')"
    ).fetchall()
    print(f"Found {len(rows)} memories to process")

    # Get current edge count before we start
    before = db.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]

    created = 0
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        contents = [r["content"] for r in batch]
        ids = [r["id"] for r in batch]

        # Generate embeddings for batch
        t0 = time.time()
        embeddings = await embed_texts(contents)
        embed_time = time.time() - t0
        print(f"  Batch {batch_start//batch_size + 1}: {len(batch)} embeddings in {embed_time:.1f}s")

        # Build embedding_id -> memory_id lookup
    id_map = {}
    for r in rows:
        eid = r["id"]  # We need the embedding_id from the full row
    # Re-fetch with embedding_id
    full_rows = db.conn.execute(
        "SELECT id, embedding_id FROM memories WHERE state IN ('active','extracted')"
    ).fetchall()
    embedding_to_memory = {r["embedding_id"]: r["id"] for r in full_rows if r["embedding_id"]}

    # For each embedding, search Qdrant and create edges
    for i, (mid, emb) in enumerate(zip(ids, embeddings)):
        if not emb:
            continue
        try:
            results = await qdrant.search(
                vector=emb,
                limit=max_links + 1,  # +1 to exclude self
                score_threshold=threshold,
            )
            for r in results:
                # Qdrant returns point_id (which is embedding_id). Map to memory id.
                target_memory_id = embedding_to_memory.get(r["id"])
                if not target_memory_id or target_memory_id == mid:
                    continue
                db.add_edge(
                    source_id=mid,
                    target_id=target_memory_id,
                    relation_type=_relation_type(r["score"]),
                    weight=r["score"],
                )
                created += 1
        except Exception as e:
            print(f"    Error for {mid[:12]}: {e}")
            continue

        # Commit after each batch
        db.conn.commit()
        pct = min(100, (batch_start + batch_size) * 100 // len(rows))
        print(f"  Progress: {pct}% | {created} edges created")

    after = db.conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]
    print(f"\nDone. Edges: {before} -> {after} (created {after - before})")


if __name__ == "__main__":
    asyncio.run(backfill())
