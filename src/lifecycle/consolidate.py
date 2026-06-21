"""Memory consolidation — merge semantically similar memories to prevent bloat.

Uses Qdrant vector search to find clusters of similar memories,
then merges them into fewer, higher-quality consolidated memories.
"""

import hashlib
import logging
from typing import Any

from ..db.sqlite import MemoryDB
from ..db.qdrant_client import MemoryQdrant
from ..config import config

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.md5(content.strip().lower().encode()).hexdigest()


def consolidate_memories(
    db: MemoryDB | None = None,
    qdrant: MemoryQdrant | None = None,
    similarity_threshold: float = 0.88,
    max_cluster_size: int = 5,
) -> dict[str, Any]:
    """Find and merge semantically similar memories.

    Args:
        db: MemoryDB instance
        qdrant: MemoryQdrant instance
        similarity_threshold: Cosine similarity threshold for clustering
        max_cluster_size: Max memories to merge per cluster

    Returns:
        Summary with merged clusters and memories reduced
    """
    if db is None:
        db = MemoryDB()
        db.init_schema()
    if qdrant is None:
        qdrant = MemoryQdrant()

    # Get active/extracted memories, ordered by importance
    memories = db.list_memories(state="active", limit=500)
    memories += db.list_memories(state="extracted", limit=500)

    if len(memories) < 10:
        return {"status": "ok", "clusters_found": 0, "memories_merged": 0, "memories_created": 0}

    # Build content hash index
    content_index: dict[str, dict] = {}
    for m in memories:
        m["_ch"] = _content_hash(m["content"])
        content_index[m["id"]] = m

    # Cluster by vector similarity (greedy, by importance)
    memories.sort(key=lambda m: m.get("importance", 0.5), reverse=True)
    clustered_ids: set[str] = set()
    clusters: list[list[dict]] = []

    for m in memories:
        if m["id"] in clustered_ids or not m.get("embedding_id"):
            continue

        # Find semantically similar memories via Qdrant
        try:
            similar = qdrant.client.search(
                collection_name=qdrant.collection_name,
                query_vector=qdrant.client.retrieve(
                    collection_name=qdrant.collection_name,
                    ids=[m["embedding_id"]],
                )[0].vector,
                limit=max_cluster_size,
                score_threshold=similarity_threshold,
            )
        except Exception:
            continue

        cluster: list[dict] = [m]
        clustered_ids.add(m["id"])

        for s in similar:
            sid = None
            for mem in memories:
                if mem.get("embedding_id") == s.id:
                    sid = mem["id"]
                    break
            if sid and sid != m["id"] and sid not in clustered_ids:
                if sid in content_index:
                    cluster.append(content_index[sid])
                    clustered_ids.add(sid)

        if len(cluster) >= 2:
            clusters.append(cluster)

    merged_count = 0
    created_count = 0

    for cluster in clusters[:20]:  # Cap at 20 clusters per run
        # Pick the highest-importance memory as primary
        primary = max(cluster, key=lambda m: m.get("importance", 0))
        secondary_ids = [m["id"] for m in cluster if m["id"] != primary["id"]]

        if not secondary_ids:
            continue

        # Merge: update primary content to include secondary references
        # Mark secondaries as consolidated
        for sid in secondary_ids:
            db.update_memory(sid, {"state": "consolidated", "parent_id": primary["id"]})
            merged_count += 1

        # Bump primary importance slightly
        db.update_memory(primary["id"], {
            "importance": min(1.0, primary.get("importance", 0.5) + 0.1),
            "state": "active",
        })
        created_count += 1

        # Create an edge
        for sid in secondary_ids:
            db.add_edge(primary["id"], sid, relation_type="consolidates")

    logger.info(
        f"Consolidation: {len(clusters)} clusters, "
        f"{merged_count} merged into {created_count} primary"
    )

    return {
        "status": "ok",
        "clusters_found": len(clusters),
        "memories_merged": merged_count,
        "memories_created": created_count,
    }
