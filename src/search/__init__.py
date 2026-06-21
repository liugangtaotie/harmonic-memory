"""Unified search — searches the structured memories table (SQLite) + optional Qdrant vector.

No longer scans raw .jsonl / .md files directly.
All results come from extracted, structured memories.
"""

import json
import os
import subprocess
import time
from typing import Any

from ..db.sqlite import MemoryDB


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _search_memories_table(query: str, max_hits: int = 30) -> list[dict]:
    """Search the memories table via FTS5 keyword search, ordered by created_at DESC."""
    from ..db.sqlite import MemoryDB
    db = MemoryDB()
    db.init_schema()

    # Try FTS5 first
    safe_query = " ".join(f'"{term}"' for term in query.split() if len(term) > 1)
    results = []
    if safe_query:
        try:
            rows = db.conn.execute(
                """SELECT m.* FROM memories m
                   JOIN memories_fts fts ON m.rowid = fts.rowid
                   WHERE memories_fts MATCH :query
                   ORDER BY m.created_at DESC
                   LIMIT :limit""",
                {"query": safe_query, "limit": max_hits},
            ).fetchall()
            results = [db._row_to_dict(r) for r in rows]
        except Exception as e:
            # FTS5 match may fail on special chars; fall through to LIKE
            pass

    # Fallback: LIKE search if FTS5 returned nothing
    if not results:
        like_terms = [f"%{t}%" for t in query.split() if len(t) > 1]
        if like_terms:
            conditions = " OR ".join(["(content LIKE ? OR summary LIKE ?)" for _ in like_terms])
            params = []
            for t in like_terms:
                params.extend([t, t])
            rows = db.conn.execute(
                f"""SELECT * FROM memories WHERE {conditions}
                    ORDER BY created_at DESC LIMIT ?""",
                params + [max_hits],
            ).fetchall()
            results = [db._row_to_dict(r) for r in rows]

    return results


def _search_mem0_qdrant(query: str, max_hits: int = 15) -> list[dict]:
    """Search mem0's Qdrant collection. Falls back to helper script."""
    results = []
    helper = _expand("~/memoryfederation/mem0_search_helper.py")
    if not os.path.exists(helper):
        return results
    try:
        result = subprocess.run(
            ["python3", helper, query],
            capture_output=True, text=True, timeout=25,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if "error" not in d:
                        results.append(d)
                except json.JSONDecodeError:
                    pass
    except (subprocess.TimeoutExpired, Exception):
        pass
    return results[:max_hits]


def _format_memory_hit(m: dict) -> dict:
    """Convert a memory row dict into a unified search hit."""
    metadata = m.get("metadata", "{}")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    return {
        "source_type": "structured_memory",
        "score": round(m.get("importance", 0.5) * 100),
        "preview": m.get("summary") or m.get("content", "")[:280],
        "file": None,
        "filename": None,
        "timestamp": m.get("created_at"),
        "source_name": m.get("source", "unknown"),
        "source_table": "memories",
        "row_data": {
            "id": m.get("id"),
            "type": m.get("type"),
            "content": m.get("content", "")[:400],
            "source": m.get("source"),
            "importance": m.get("importance"),
            "confidence": m.get("confidence"),
            "keywords": metadata.get("keywords", []),
        },
    }


def unified_search(query: str, max_per_source: int = 30, offset: int = 0) -> dict[str, Any]:
    """Search structured memories (SQLite FTS5) + mem0 vector.

    Returns:
        dict with query, results (list of structured hits),
        counts per source, total, and latency.
    """
    t0 = time.time()
    total_count = 0

    # 1. Structured memories (SQLite)
    if query.strip():
        memory_hits = _search_memories_table(query, max_hits=max_per_source)
        total_count = len(memory_hits)
    else:
        # Empty query -> return all recent memories
        from ..db.sqlite import MemoryDB
        db = MemoryDB()
        db.init_schema()
        total_count = db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        rows = db.conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (max_per_source, offset),
        ).fetchall()
        memory_hits = [db._row_to_dict(r) for r in rows]

    # 2. mem0 vector (Qdrant) as enrichment
    mem0_hits = _search_mem0_qdrant(query, max_hits=10)

    # ── Unify format ──
    all_results = []

    for m in memory_hits:
        all_results.append(_format_memory_hit(m))

    for h in mem0_hits:
        all_results.append({
            "source_type": "mem0_vector",
            "score": h.get("score", 0),
            "preview": h.get("preview", h.get("memory", ""))[:280],
            "file": None,
            "filename": None,
            "timestamp": h.get("timestamp") or h.get("created_at"),
            "source_name": "mem0",
            "source_table": "mem0",
            "row_data": h,
        })

    # Sort: timestamp descending, then score
    def sort_key(r):
        ts = 0
        raw_ts = r.get("timestamp")
        if raw_ts:
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        sc = r.get("score", 0) or 0
        return (ts, sc)

    all_results.sort(key=sort_key, reverse=True)

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "query": query,
        "results": all_results,
        "total": total_count or len(all_results),
        "sources": {
            "structured_memory": len(memory_hits),
            "mem0_vector": len(mem0_hits),
        },
        "latency_ms": latency_ms,
    }
