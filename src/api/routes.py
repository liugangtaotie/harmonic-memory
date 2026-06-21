"""FastAPI route definitions for Harmonic Memory."""

import time

from fastapi import APIRouter, HTTPException, Query

from ..ingest import ingest as ingest_pipeline
from ..db.sqlite import MemoryDB
from ..db.qdrant_client import MemoryQdrant
from ..search import unified_search as unified_search_fn
from ..sources.session_watcher import ingest_recent_sessions
from ..lifecycle.decay import run_decay_cycle
from ..lifecycle.consolidate import consolidate_memories
from ..neural import spread_activate, reinforce_edges
from .schemas import (
    IngestRequest, IngestResponse,
    MemoryResponse, MemoryUpdate,
    SearchRequest, SearchResponse,
    ProfileEntry, ProfileUpdate,
    HealthResponse, StatsResponse,
    UnifiedSearchHit, UnifiedSearchResponse,
)

router = APIRouter(prefix="/api/v1", tags=["harmonic-memory"])

# Global instances — initialized in server.py
db: MemoryDB | None = None
qdrant: MemoryQdrant | None = None
start_time: float = time.time()


def get_db() -> MemoryDB:
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db


def get_qdrant() -> MemoryQdrant:
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")
    return qdrant


# ─── Ingestion ───

@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(req: IngestRequest):
    """Ingest raw text — extract, classify, embed, dedup, store."""
    result = await ingest_pipeline(
        text=req.text,
        source=req.source,
        source_ref=req.source_ref,
        db=get_db(),
        qdrant=get_qdrant(),
        use_fallback=req.use_fallback,
    )
    return IngestResponse(**result)


# ─── Search ───

@router.get("/search", response_model=SearchResponse)
async def search_memories(
    q: str = Query(default="", description="Search query (empty = all recent)"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    type: str | None = Query(default=None),
    source: str | None = Query(default=None),
):
    """Hybrid search across memories (keyword + semantic)."""
    t0 = time.time()
    database = get_db()
    qdrant_store = get_qdrant()

    # Keyword search via FTS5
    keyword_results = database.search_keyword(q, limit=limit * 2)

    # For now, keyword search is primary; vector search added in Phase 2
    results = keyword_results

    # Apply filters
    if type:
        results = [r for r in results if r.get("type") == type]
    if source:
        results = [r for r in results if r.get("source") == source]

    # Paginate
    total = len(results)
    page = results[offset:offset + limit]

    elapsed = int((time.time() - t0) * 1000)

    return SearchResponse(
        query=q,
        results=[MemoryResponse(**r) for r in page],
        total=total,
        latency_ms=elapsed,
    )


# ─── Memory CRUD ───

@router.get("/memories", response_model=list[MemoryResponse])
async def list_memories(
    type: str | None = None,
    state: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List memories with optional filters."""
    database = get_db()
    memories = database.list_memories(
        type=type, state=state, source=source,
        limit=limit, offset=offset,
    )
    return [MemoryResponse(**m) for m in memories]


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Get a single memory by ID."""
    database = get_db()
    mem = database.get_memory(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    database.record_access(memory_id)
    return MemoryResponse(**mem)


@router.get("/memories/{memory_id}/related", response_model=list[MemoryResponse])
async def get_related_memories(
    memory_id: str,
    max_depth: int = Query(default=3, ge=1, le=5),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Get related memories via spreading activation through the edge graph.

    Follows edges bidirectionally, decaying activation per hop.
    Records access on all activated memories for Hebbian reinforcement.
    """
    database = get_db()

    # Verify memory exists
    mem = database.get_memory(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Record access on the seed
    database.record_access(memory_id)

    # Spread activation through the graph
    activated = spread_activate(
        seed_id=memory_id,
        db=database,
        max_depth=max_depth,
        max_results=limit,
    )

    # Fetch full memory objects + record access for reinforcement
    results = []
    for a in activated:
        m = database.get_memory(a["memory_id"])
        if m:
            m["_activation"] = a["activation"]
            m["_depth"] = a["depth"]
            results.append(m)
            database.record_access(a["memory_id"])

    # Hebbian reinforcement: strengthen edges along activation paths
    if activated:
        try:
            chain = [memory_id] + [a["memory_id"] for a in activated[:5]]
            reinforce_edges(database, chain, boost=0.05)
        except Exception:
            pass

    return [MemoryResponse(**r) for r in results]


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(memory_id: str, update: MemoryUpdate):
    """Update memory fields."""
    database = get_db()
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    success = database.update_memory(memory_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    mem = database.get_memory(memory_id)
    return MemoryResponse(**mem)


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory and its vector."""
    database = get_db()
    qdrant_store = get_qdrant()
    mem = database.get_memory(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Delete vector
    if mem.get("embedding_id"):
        qdrant_store.delete(mem["embedding_id"])

    database.delete_memory(memory_id)
    return {"status": "deleted", "id": memory_id}


# ─── User Profile ───

@router.get("/profile", response_model=list[ProfileEntry])
async def get_profile(
    category: str | None = None,
):
    """Get user profile entries."""
    database = get_db()
    entries = database.get_profile(category=category)
    return [ProfileEntry(**e) for e in entries]


@router.put("/profile")
async def set_profile(entry: ProfileUpdate):
    """Set a user profile attribute."""
    database = get_db()
    database.set_profile(
        key=entry.key,
        value=entry.value,
        category=entry.category,
        confidence=entry.confidence,
    )
    return {"status": "ok", "key": entry.key}


# ─── Unified Search ───

@router.get("/search-unified", response_model=UnifiedSearchResponse)
async def search_unified(
    q: str = Query(default="", description="Search query (empty = all recent)"),
    source: str | None = Query(default=None, description="Filter: claude_session, files, codex, mem0_vector"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """Unified search across structured memories.

    Searches SQLite memories table (FTS5) + optional mem0 Qdrant.
    Supports pagination via offset/limit for infinite scroll.
    """
    t0 = time.time()
    # Fetch enough for the requested page
    fetch_size = min(offset + limit + 20, 300)
    raw = unified_search_fn(query=q, max_per_source=fetch_size)

    # Apply optional source/type filter
    if source:
        raw["results"] = [
            r for r in raw["results"]
            if r.get("source_type") == source
            or (r.get("row_data") or {}).get("type") == source
        ]

    total = raw.get("total", len(raw["results"]))
    # Paginate
    raw["results"] = raw["results"][offset:offset + limit]
    raw["latency_ms"] = int((time.time() - t0) * 1000)

    return UnifiedSearchResponse(
        query=q,
        results=[UnifiedSearchHit(**r) for r in raw["results"]],
        total=total,
        sources=raw["sources"],
        latency_ms=raw["latency_ms"],
    )


# ─── Health & Stats ───

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    import time as _time
    from .. import __version__

    qdrant_info = {}
    try:
        qdrant_store = get_qdrant()
        qdrant_info = qdrant_store.get_collection_info()
    except Exception as e:
        qdrant_info = {"error": str(e)}

    sqlite_info = {}
    try:
        database = get_db()
        sqlite_info = database.get_stats()
    except Exception as e:
        sqlite_info = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        version=__version__,
        qdrant=qdrant_info,
        sqlite=sqlite_info,
        uptime_seconds=_time.time() - start_time,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    database = get_db()
    qdrant_store = get_qdrant()

    db_stats = database.get_stats()
    ingestion_stats = database.get_ingestion_stats(hours=24)
    qdrant_count = qdrant_store.count()

    return StatsResponse(
        total_memories=db_stats["total_memories"],
        by_type=db_stats["by_type"],
        by_state=db_stats["by_state"],
        total_edges=db_stats["total_edges"],
        profile_attributes=db_stats["profile_attributes"],
        qdrant_vectors=qdrant_count,
        ingestion_24h=ingestion_stats,
    )


# ─── Real-time Ingestion ───

@router.post("/ingest-recent")
async def ingest_recent(
    max_files: int = 5,
):
    """Scan recent session transcript files and extract new memories.

    Called by the dashboard on load and periodically via cron.
    Only processes files that haven't been ingested yet.
    """
    database = get_db()
    result = await ingest_recent_sessions(
        db=database,
        max_files=max_files,
    )
    return result


# ─── Memory Maintenance ───

@router.post("/decay")
async def run_decay():
    """Run memory decay cycle: score, archive low-score, mark decayed."""
    database = get_db()
    result = await run_decay_cycle(db=database)
    return result


@router.post("/consolidate")
async def run_consolidate():
    """Consolidate semantically similar memories to reduce bloat."""
    database = get_db()
    qdrant_store = get_qdrant()
    result = consolidate_memories(db=database, qdrant=qdrant_store)
    return result
